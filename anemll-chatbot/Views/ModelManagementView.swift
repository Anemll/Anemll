private func forceRedownloadModel(_ model: Model) {
    print("🔄 Starting forced redownload for model: \(model.id)")
    
    // First update UI to show download is starting
    isDownloading[model.id] = true
    downloadProgress[model.id] = 0.01
    currentDownloadingFile[model.id] = "Preparing for full redownload..."
    
    // Refresh immediately to show download status
    refreshModels()
    
    // Call the modelService to force redownload
    modelService.forceRedownloadModel(
        modelId: model.id,
        fileProgress: { (file, progress) in
            DispatchQueue.main.async {
                // Update progress
                self.downloadProgress[model.id] = progress
                self.currentDownloadingFile[model.id] = file
                
                // Force refresh to update UI
                self.refreshModels()
            }
        },
        completion: { success in
            DispatchQueue.main.async {
                // Update download status
                self.isDownloading[model.id] = false
                
                if success {
                    print("✅ Model redownload completed successfully: \(model.id)")
                    // Update success message
                    self.successMessage = "Model '\(model.name)' has been successfully downloaded."
                    self.showSuccess = true
                } else {
                    print("❌ Model redownload failed: \(model.id)")
                    // Update error message
                    self.errorMessage = "Failed to download model '\(model.name)'. Please check your internet connection and try again."
                    self.showError = true
                }
                
                // Refresh model list to reflect changes
                self.refreshModels()
            }
        }
    )
} 