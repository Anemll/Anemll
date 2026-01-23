import Foundation
import Metal
import CoreVideo
import CoreML
import IOSurface

/// Metal-based argmax for processing logits on GPU
/// Processes all 16 IOSurface-backed buffers in a single GPU dispatch
public class MetalArgmax {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let argmaxPipeline: MTLComputePipelineState

    // Pre-allocated buffers
    private var partialResultsBuffer: MTLBuffer?  // 16 x (maxVal, maxIdx)
    private let chunkSize = 16384
    private let numChunks = 16

    public init?() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not available")
            return nil
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            print("Failed to create Metal command queue")
            return nil
        }
        self.commandQueue = queue

        // Single kernel that processes one chunk and writes partial result
        // Uses threadgroup reduction for efficiency
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct PartialResult {
            float maxVal;
            int maxIdx;
        };

        // Parallel reduction argmax within a single chunk
        kernel void argmax_chunk(
            device const half* input [[buffer(0)]],
            device PartialResult* results [[buffer(1)]],
            constant uint& chunkSize [[buffer(2)]],
            constant uint& globalOffset [[buffer(3)]],
            constant uint& skipFirst [[buffer(4)]],
            constant uint& chunkIndex [[buffer(5)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            // Shared memory for reduction
            threadgroup float sharedMax[256];
            threadgroup int sharedIdx[256];

            // Each thread finds local max over its portion
            float localMax = -INFINITY;
            int localIdx = 0;

            uint start = (chunkIndex == 0 && skipFirst > 0) ? skipFirst : 0;
            uint elementsPerThread = (chunkSize + tgSize - 1) / tgSize;
            uint myStart = tid * elementsPerThread;
            uint myEnd = min(myStart + elementsPerThread, chunkSize);

            // Adjust for skipFirst
            if (myStart < start) myStart = start;

            for (uint i = myStart; i < myEnd; i++) {
                float val = float(input[i]);
                if (val > localMax) {
                    localMax = val;
                    localIdx = int(globalOffset + i);
                }
            }

            // Store in shared memory
            sharedMax[tid] = localMax;
            sharedIdx[tid] = localIdx;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Parallel reduction
            for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    if (sharedMax[tid + stride] > sharedMax[tid]) {
                        sharedMax[tid] = sharedMax[tid + stride];
                        sharedIdx[tid] = sharedIdx[tid + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Thread 0 writes final result for this chunk
            if (tid == 0) {
                results[chunkIndex].maxVal = sharedMax[0];
                results[chunkIndex].maxIdx = sharedIdx[0];
            }
        }
        """

        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)

            guard let argmaxFunc = library.makeFunction(name: "argmax_chunk") else {
                print("Failed to create Metal function")
                return nil
            }

            self.argmaxPipeline = try device.makeComputePipelineState(function: argmaxFunc)
        } catch {
            print("Failed to create Metal pipeline: \(error)")
            return nil
        }

        // Pre-allocate results buffer: 16 chunks x (float maxVal + int maxIdx)
        partialResultsBuffer = device.makeBuffer(length: numChunks * 8, options: .storageModeShared)
    }

    /// Find argmax across all 16 IOSurface-backed logits chunks
    public func findArgmax(
        backings: [String: MLMultiArray],
        splitCount: Int,
        vocabSize: Int,
        filterFirst: Bool = false
    ) throws -> Int {
        guard let resultsBuffer = partialResultsBuffer else {
            throw MetalError.bufferNotAvailable
        }

        // First, lock all IOSurfaces and create Metal buffers
        var ioSurfaces: [IOSurfaceRef] = []
        var metalBuffers: [MTLBuffer] = []

        for i in 0..<splitCount {
            let logitsKey = "logits\(i + 1)"
            guard let logitsPart = backings[logitsKey],
                  let pixelBuffer = logitsPart.pixelBuffer else {
                throw MetalError.noIOSurface
            }

            guard let ioSurface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else {
                throw MetalError.noIOSurface
            }

            // Lock IOSurface for reading
            IOSurfaceLock(ioSurface, .readOnly, nil)
            ioSurfaces.append(ioSurface)

            let baseAddress = IOSurfaceGetBaseAddress(ioSurface)
            let bytesPerRow = IOSurfaceGetBytesPerRow(ioSurface)
            let height = IOSurfaceGetHeight(ioSurface)
            let totalBytes = bytesPerRow * height

            guard let metalBuffer = device.makeBuffer(
                bytesNoCopy: baseAddress,
                length: totalBytes,
                options: .storageModeShared,
                deallocator: nil
            ) else {
                // Unlock already locked surfaces before throwing
                for surface in ioSurfaces {
                    IOSurfaceUnlock(surface, .readOnly, nil)
                }
                throw MetalError.bufferCreationFailed
            }

            metalBuffers.append(metalBuffer)
        }

        // Now create command buffer and encode all chunks
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            for surface in ioSurfaces {
                IOSurfaceUnlock(surface, .readOnly, nil)
            }
            throw MetalError.commandBufferFailed
        }

        for i in 0..<splitCount {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                for surface in ioSurfaces {
                    IOSurfaceUnlock(surface, .readOnly, nil)
                }
                throw MetalError.encoderFailed
            }

            encoder.setComputePipelineState(argmaxPipeline)
            encoder.setBuffer(metalBuffers[i], offset: 0, index: 0)
            encoder.setBuffer(resultsBuffer, offset: 0, index: 1)

            var chunkSizeVal = UInt32(chunkSize)
            var globalOffset = UInt32(i * chunkSize)
            var skipFirst = UInt32((filterFirst && i == 0) ? 2 : 0)
            var chunkIndex = UInt32(i)

            encoder.setBytes(&chunkSizeVal, length: 4, index: 2)
            encoder.setBytes(&globalOffset, length: 4, index: 3)
            encoder.setBytes(&skipFirst, length: 4, index: 4)
            encoder.setBytes(&chunkIndex, length: 4, index: 5)

            let threadGroupSize = 256
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Now unlock all IOSurfaces
        for surface in ioSurfaces {
            IOSurfaceUnlock(surface, .readOnly, nil)
        }

        if let error = commandBuffer.error {
            throw MetalError.executionFailed(error)
        }

        // Find global max from partial results on CPU (fast for 16 values)
        let resultsPtr = resultsBuffer.contents()
        var globalMaxVal: Float = -.infinity
        var globalMaxIdx: Int = 0

        for i in 0..<splitCount {
            let offset = i * 8
            let maxVal = resultsPtr.load(fromByteOffset: offset, as: Float.self)
            let maxIdx = resultsPtr.load(fromByteOffset: offset + 4, as: Int32.self)

            if maxVal > globalMaxVal {
                globalMaxVal = maxVal
                globalMaxIdx = Int(maxIdx)
            }
        }

        return globalMaxIdx
    }
}

enum MetalError: Error {
    case bufferNotAvailable
    case commandBufferFailed
    case encoderFailed
    case noIOSurface
    case bufferCreationFailed
    case executionFailed(Error)
}
