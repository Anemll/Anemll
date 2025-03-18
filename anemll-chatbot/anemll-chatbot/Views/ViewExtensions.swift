// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ViewExtensions.swift

import SwiftUI
import Combine

// Extension to help with consistently displaying model loading progress
extension View {
    @MainActor
    @ViewBuilder
    func withModelLoadingIndicator(inferenceService: InferenceService? = nil) -> some View {
        let service = inferenceService ?? InferenceService.shared
        
        ZStack(alignment: .top) {
            self
            
            if service.isLoadingModel {
                VStack {
                    ModelLoadingView()
                        .padding(.horizontal)
                        .padding(.top, 4)
                    
                    Spacer()
                }
            }
        }
    }
}

// This version allows directly embedding the ModelLoadingView in any view
// without import issues since it's locally defined here
extension View {
    @MainActor
    @ViewBuilder
    func modelLoadingProgress() -> some View {
        ZStack(alignment: .bottom) {
            self
            
            if InferenceService.shared.isLoadingModel {
                ModelLoadingView()
                    .padding()
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
    }
} 