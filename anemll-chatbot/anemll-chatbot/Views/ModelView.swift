import SwiftUI
import Combine

#if targetEnvironment(macCatalyst)
import AppKit
#endif

// Custom view modifier to handle ESC key presses
struct EscKeyHandler: ViewModifier {
    let onEscKeyPressed: () -> Void
    
    func body(content: Content) -> some View {
        content
            .onAppear {
                NotificationCenter.default.addObserver(
                    forName: UIApplication.willResignActiveNotification,
                    object: nil,
                    queue: .main
                ) { _ in
                    // This can catch ESC key in some cases 
                    onEscKeyPressed()
                }
                
                #if targetEnvironment(macCatalyst)
                // For Mac Catalyst, we would add keyboard monitoring here, but AppKit integration
                // requires additional setup we can't do in this simple modifier
                // The interactiveDismissDisabled(false) will handle ESC key for us in sheets
                #endif
            }
    }
}

extension View {
    func onEscKeyPress(action: @escaping () -> Void) -> some View {
        self.modifier(EscKeyHandler(onEscKeyPressed: action))
    }
}

struct ModelView: View {
    @ObservedObject private var modelService = ModelService.shared
    @State private var showingAddModelSheet = false
    @State private var searchText = ""
    @State private var thinkingMode = false
    @State private var allowLongGeneration = false
    @State private var selectedModelId: String? = nil
    @State private var showSelectionMessage = false
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        List {
            // Add selection message banner if a model is selected
            if showSelectionMessage, let modelId = selectedModelId, let model = modelService.getModel(for: modelId) {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text("Model \"\(model.name)\" was selected.")
                                .font(.headline)
                            Spacer()
                            Button {
                                showSelectionMessage = false
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundColor(.gray)
                            }
                        }
                        
                        Text("Please click Load to load it to memory.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Button("Load Model") {
                            modelService.loadModel(model)
                            // Keep window open, but hide the message
                            showSelectionMessage = false
                        }
                        .buttonStyle(.borderedProminent)
                        .padding(.top, 4)
                    }
                    .padding(.vertical, 4)
                }
            }
            
            Section(header: Text("Available Models")) {
                ForEach(filteredModels) { model in
                    ModelRow(model: model, onModelSelected: { selectedModel in
                        // Instead of loading and closing, just show a message
                        selectedModelId = selectedModel.id
                        showSelectionMessage = true
                    })
                }
            }
            
            Button(action: {
                showingAddModelSheet = true
            }) {
                Label("Add Custom Model", systemImage: "plus.circle")
            }
            
            Section(header: Text("Generation Settings")) {
                Toggle("Enable Thinking Mode", isOn: $thinkingMode)
                    .onChange(of: thinkingMode) { oldValue, newValue in
                        Task {
                            InferenceService.shared.setThinkingMode(newValue)
                        }
                    }
                
                Text("Shows reasoning process in <think> tags")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Divider()
                
                Toggle("Enable Long Generation", isOn: $allowLongGeneration)
                
                Text("Allows generation up to 4x the model's context length with automatic window shifting")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .searchable(text: $searchText, prompt: "Search models")
        .navigationTitle("Models")
        .sheet(isPresented: $showingAddModelSheet) {
            AddModelView(isPresented: $showingAddModelSheet)
        }
        // Add keyboard handling for ESC key
        .onAppear {
            print("DEBUG: ModelView appeared")
            // Load current thinking mode setting
            Task {
                thinkingMode = InferenceService.shared.isThinkingModeEnabled()
            }
            // Debug current generation settings
            let longGenLimit = maxGenerationTokens
            print("DEBUG - Generation limit with long generation \(allowLongGeneration ? "enabled" : "disabled"): \(longGenLimit)")
        }
        // Allow clicking outside to dismiss
        .interactiveDismissDisabled(false)
        // Add ESC key handler
        .onEscKeyPress {
            dismiss()
        }
    }
    
    private var filteredModels: [Model] {
        if searchText.isEmpty {
            return modelService.getAvailableModels()
        } else {
            return modelService.getAvailableModels().filter { model in
                model.name.localizedCaseInsensitiveContains(searchText) ||
                model.description.localizedCaseInsensitiveContains(searchText)
            }
        }
    }
    
    private var maxGenerationTokens: Int {
        allowLongGeneration ? 4096 : 1024
    }
}

struct ModelRow: View {
    let model: Model
    @ObservedObject private var modelService = ModelService.shared
    @State private var isDownloading = false
    @State private var downloadProgress: Double = 0
    @State private var downloadStatus: String = ""
    var onModelSelected: ((Model) -> Void)?
    
    var body: some View {
        HStack {
            // Main row content - wrapped in a button
            Button(action: {
                if model.isDownloaded {
                    // Instead of automatically loading, just mark as selected
                    onModelSelected?(model)
                } else {
                    startDownload()
                }
            }) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(model.name)
                            .font(.headline)
                            .foregroundColor(isSelected ? .blue : .primary)
                        
                        Text(model.description)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                        
                        // Add source URL display
                        Text(ModelService.shared.getDisplayURL(for: model))
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                        
                        HStack {
                            if model.isDownloaded {
                                Label("Downloaded", systemImage: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .font(.caption)
                            } else if isDownloading {
                                HStack {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .blue))
                                        .scaleEffect(0.7)
                                    
                                    Text(downloadStatus)
                                        .font(.caption)
                                        .foregroundColor(.blue)
                                }
                            } else {
                                Label("Not Downloaded", systemImage: "arrow.down.circle")
                                    .foregroundColor(.blue)
                                    .font(.caption)
                            }
                            
                            Spacer()
                            
                            Text(formatFileSize(model.size))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Spacer()
                    
                    if !model.isDownloaded {
                        Image(systemName: "arrow.down.circle")
                            .foregroundColor(.blue)
                            .font(.title2)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(PlainButtonStyle())
            
            // Place action buttons outside the parent button
            if model.isDownloaded {
                HStack(spacing: 12) {
                    // Play button
                    Button(action: {
                        modelService.loadModel(model)
                    }) {
                        Image(systemName: "play.circle")
                            .foregroundColor(.blue)
                            .font(.title2)
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    // Reload button
                    Button(action: {
                        // First cancel any current loading
                        if InferenceService.shared.isLoadingModel {
                            InferenceService.shared.cancelModelLoading(reason: .startingNewModel)
                            
                            // Wait a moment to ensure cancellation is complete
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                modelService.loadModel(model)
                            }
                        } else {
                            // If nothing is loading, just load the model
                            modelService.loadModel(model)
                        }
                    }) {
                        Image(systemName: "arrow.clockwise.circle")
                            .foregroundColor(.blue)
                            .font(.title2)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                .padding(.leading, 8)
            }
        }
        .contextMenu {
            if model.isDownloaded {
                Button(action: {
                    modelService.loadModel(model)
                }) {
                    Label("Load Model", systemImage: "play.circle")
                }
                
                Button(action: {
                    // Reload model
                    if InferenceService.shared.isLoadingModel {
                        InferenceService.shared.cancelModelLoading(reason: .startingNewModel)
                        
                        // Wait a moment to ensure cancellation is complete
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                            modelService.loadModel(model)
                        }
                    } else {
                        modelService.loadModel(model)
                    }
                }) {
                    Label("Reload Model", systemImage: "arrow.clockwise.circle")
                }
            } else {
                Button(action: {
                    startDownload()
                }) {
                    Label("Download", systemImage: "arrow.down.circle")
                }
            }
            
            if isDownloading {
                Button(action: {
                    cancelDownload()
                }) {
                    Label("Cancel Download", systemImage: "xmark.circle")
                }
            }
        }
        .onAppear {
            checkDownloadStatus()
        }
    }
    
    private var isSelected: Bool {
        modelService.isModelSelected(model)
    }
    
    private func startDownload() {
        isDownloading = true
        downloadStatus = "Starting..."
        
        modelService.downloadModel(modelId: model.id, fileProgress: { status, progress in
            self.downloadStatus = status
            self.downloadProgress = progress
        }) { success in
            self.isDownloading = false
            if !success {
                self.downloadStatus = "Failed"
            }
        }
    }
    
    private func cancelDownload() {
        modelService.cancelDownload(modelId: model.id)
        isDownloading = false
        downloadStatus = "Cancelled"
    }
    
    private func checkDownloadStatus() {
        if let status = modelService.getCurrentDownloadingFile(for: model.id) {
            isDownloading = true
            downloadStatus = status
            
            if let progress = modelService.getDownloadProgress(for: model.id) {
                downloadProgress = progress
            }
        } else {
            isDownloading = false
        }
    }
    
    private func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
}

struct AddModelView: View {
    @Binding var isPresented: Bool
    @State private var name: String = ""
    @State private var description: String = ""
    @State private var downloadURL: String = ""
    @State private var showingAlert = false
    @State private var alertMessage = ""
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model Details")) {
                    TextField("Name", text: $name)
                    TextField("Description", text: $description)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        TextField("Download URL (required)", text: $downloadURL)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                            .keyboardType(.URL)
                            .onChange(of: downloadURL) { oldValue, newValue in
                                print("URL changed to: '\(newValue)'")
                            }
                        
                        Text("Enter a Hugging Face URL like: https://huggingface.co/owner/repo")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Section {
                    Button("Add Model") {
                        addModel()
                    }
                    .disabled(name.isEmpty || downloadURL.isEmpty)
                }
            }
            .navigationTitle("Add Custom Model")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
            }
            .alert(isPresented: $showingAlert) {
                Alert(
                    title: Text("Error"),
                    message: Text(alertMessage),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
        // Add ESC key and click outside support
        .interactiveDismissDisabled(false)
        // Add ESC key handler
        .onEscKeyPress {
            isPresented = false 
        }
    }
    
    private func addModel() {
        // Debug logging to check URL value
        print("DEBUG - Adding custom model with:")
        print("Name: \(name)")
        print("Description: \(description)")
        print("URL before passing to service: '\(downloadURL)'")
        
        ModelService.shared.addCustomModel(
            name: name.isEmpty ? nil : name,
            description: description.isEmpty ? nil : description,
            downloadURL: downloadURL
        ) { success, error in
            if success {
                print("DEBUG - Model added successfully with URL: '\(downloadURL)'")
                isPresented = false
            } else if let error = error {
                print("DEBUG - Failed to add model: \(error)")
                alertMessage = error
                showingAlert = true
            }
        }
    }
}

struct ModelView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            ModelView()
        }
    }
} 