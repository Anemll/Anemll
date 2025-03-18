// AcknowledgmentsView.swift
import SwiftUI

struct AcknowledgmentsView: View {
    let acknowledgments: [Acknowledgment]

    var body: some View {
        List(acknowledgments) { acknowledgment in
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 8) {
                    // Check if this is the ANEMLL acknowledgment and add a special website link
                    if acknowledgment.name == "ANEMLL" {
                        Link("Visit www.anemll.com", destination: URL(string: "https://www.anemll.com")!)
                            .font(.caption)
                            .foregroundColor(.blue)
                            .padding(.bottom, 4)
                    }
                    
                    Text(acknowledgment.licenseText)
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            } label: {
                Text(acknowledgment.name)
                    .font(.headline)
            }
        }
        .navigationTitle("Acknowledgments")
    }
}

#Preview {
    AcknowledgmentsView(acknowledgments: acknowledgments)
}