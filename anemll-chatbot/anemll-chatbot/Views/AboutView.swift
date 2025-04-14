// AboutView.swift
import SwiftUI

struct AboutView: View {
    @State private var showingAcknowledgments = false
    
    private var versionString: String {
        if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
            return "Version \(version)"
        }
        return "Version Unknown"
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("anemll-chatbot")
                .font(.largeTitle)
                .fontWeight(.bold)
            Text(versionString)
                .font(.title3)
            Text("Copyright © 2025 Anemll")
                .font(.subheadline)
            
            Link("Visit www.anemll.com", destination: URL(string: "https://www.anemll.com")!)
                .font(.subheadline)
                .foregroundColor(.blue)
            
            Button("Acknowledgments") {
                showingAcknowledgments = true
            }
            .font(.body)
            .padding()
            .background(Color.blue.opacity(0.1))
            .cornerRadius(8)
        }
        .sheet(isPresented: $showingAcknowledgments) {
            AcknowledgmentsView(acknowledgments: acknowledgments)
        }
        .navigationTitle("About")
    }
}

#Preview {
    NavigationStack {
        AboutView()
    }
}