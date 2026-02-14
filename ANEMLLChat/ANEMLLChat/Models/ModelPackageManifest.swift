//
//  ModelPackageManifest.swift
//  ANEMLLChat
//
//  Package manifest contract for macOS -> iOS model transfer
//

import Foundation

struct ModelPackageManifest: Codable, Sendable {
    let formatVersion: Int
    let modelName: String
    let modelId: String?
    let modelRootPath: String?
    let minAppVersion: String?
    let files: [ModelPackageFileEntry]
}

struct ModelPackageFileEntry: Codable, Sendable {
    let path: String
    let sha256: String
    let sizeBytes: Int64?
}
