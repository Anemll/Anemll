let modelInfoContent = String(yamlContent[modelInfoStart...]) 

// This function is already defined in ModelManagementView.swift as an extension to ModelService
// Uncomment only if needed for a specific version
/* 
func calculateActualModelSize(modelId: String) -> Int {
    // This method is implemented in ModelManagementView
    // We're just forwarding the call to avoid duplication
    return ModelManagementView().calculateActualModelSize(modelId: modelId)
} 
*/ 