// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ModelLoadingView.swift

import SwiftUI
import Combine

struct ModelLoadingView: View {
    @ObservedObject private var inferenceService = InferenceService.shared
    @State private var progress: Double = 0.0
    @State private var status: String = ""
    @State private var progressString: String = "0%"
    @State private var refreshTrigger = UUID()
    @State private var cancellables = Set<AnyCancellable>()
    
    var body: some View {
        Group {
            if inferenceService.isLoadingModel {
                VStack(spacing: 4) {
                    HStack {
                        // Cancel button
                        Button(action: {
                            print("⛔️ User cancelled model loading")
                            print("⛔️ DEBUG: inferenceService type: \(type(of: inferenceService))")
                            // Debug current model loader state
                            inferenceService.debugModelLoaderState()
                            
                            // First set a local flag to update UI immediately
                            status = "Cancelling..."
                            
                            // Use Task to handle cancellation in background
                            Task { @MainActor in
                                print("⛔️ Inside cancel button Task - Before calling forceCancel")
                                // Call the enhanced cancellation method
                                inferenceService.forceCancel()
                                
                                print("⛔️ After forceCancel call - Updating UI")
                                // Update UI to show cancellation is in progress
                                progress = 0.0
                                progressString = "0%"
                                status = "Cancelled"
                                
                                // Debug state after cancellation
                                print("⛔️ Model loader state after cancellation:")
                                inferenceService.debugModelLoaderState()
                                
                                // Force refresh UI
                                refreshTrigger = UUID()
                                print("⛔️ Cancel button Task complete")
                            }
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.red)
                                .font(.system(size: 16))
                        }
                        .buttonStyle(BorderlessButtonStyle())
                        .padding(.trailing, 4)
                        
                        // Loading spinner
                        ProgressView()
                            .scaleEffect(0.8)
                        
                        // Progress text
                        Text(progressString)
                            .font(.caption)
                            .padding(.leading, 4)
                            .frame(width: 50, alignment: .leading)
                        
                        Spacer()
                        
                        // Status text
                        Text(status)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    
                    // Progress bar
                    ProgressView(value: progress)
                        .progressViewStyle(LinearProgressViewStyle())
                }
                .padding(8)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .id(refreshTrigger) // Force refresh when triggered
                .onAppear {
                    setupObservers()
                    updateData()
                    
                    // Log initial state
                    print("📊 ModelLoadingView appeared with initial state:")
                    print("  - isModelLoaded: \(inferenceService.isModelLoaded)")
                    print("  - isLoadingModel: \(inferenceService.isLoadingModel)")
                    print("  - loadingProgress: \(inferenceService.loadingProgress)")
                    print("  - loadingProgressString: \(inferenceService.loadingProgressString)")
                    print("  - loadingStatus: \(inferenceService.loadingStatus)")
                }
            } else {
                // Empty view when not loading
                EmptyView()
            }
        }
        .onChange(of: inferenceService.loadingProgress) { _, newValue in
            print("📊 Progress changed to: \(newValue)")
            progress = newValue
            progressString = "\(Int(newValue * 100))%"
        }
        .onChange(of: inferenceService.loadingStatus) { _, newValue in
            print("📊 Status changed to: \(newValue)")
            status = newValue
        }
        .onReceive(Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()) { _ in
            updateData()
        }
    }
    
    private func setupObservers() {
        // Clear existing observers
        cancellables.removeAll()
        
        // Explicit notification monitoring
        NotificationCenter.default.publisher(for: Notification.Name("LoadingProgressChanged"))
            .sink { _ in
                updateData()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: Notification.Name("ModelLoadingProgressUpdated"))
            .sink { _ in
                updateData()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: Notification.Name("ExplicitProgressUpdate"))
            .sink { notification in
                if let newProgress = notification.object as? Double {
                    progress = newProgress
                    progressString = "\(Int(newProgress * 100))%"
                }
                updateData()
            }
            .store(in: &cancellables)
    }
    
    private func updateData() {
        // Directly fetch the latest data from inferenceService
        progress = inferenceService.getProgressValue()
        status = inferenceService.getLoadingStatus()
        progressString = inferenceService.getProgressString()
        
        // Force refresh by changing ID if needed
        if progressString != "\(Int(progress * 100))%" {
            progressString = "\(Int(progress * 100))%"
            refreshTrigger = UUID()
        }
        
        // Log current values for debugging (only log every few updates to reduce spam)
        if Int.random(in: 0...10) == 0 {
            print("ModelLoadingView update: Progress=\(progress) (\(progressString)), Status=\(status)")
        }
    }
}

#Preview {
    VStack {
        Text("Model Loading Preview")
        ModelLoadingView()
            .padding()
    }
} 