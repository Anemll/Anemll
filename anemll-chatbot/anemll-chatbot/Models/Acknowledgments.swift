import SwiftUI

struct Acknowledgment: Identifiable {
    let id = UUID()
    let name: String
    let licenseText: String
}

// Sample acknowledgment data (replace license texts with actual ones)
let acknowledgments = [
        Acknowledgment(
        name: "ANEMLL",
        licenseText: """
        Copyright (c) 2025 Anemll
        Website: www.anemll.com
        
        Licensed under the MIT License
        [Full MIT text here]
        """
    ),
    Acknowledgment(
        name: "Swift Argument Parser",
        licenseText: """
        Copyright (c) 2020 Apple Inc. and the Swift Argument Parser project authors.
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
        [Full Apache 2.0 text here]
        """
    ),
    Acknowledgment(
        name: "Yams",
        licenseText: """
        Copyright (c) 2016 JP Simard
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        [Full MIT text here]
        """
    ),
    Acknowledgment(
        name: "PathKit",
        licenseText: """
        Copyright (c) 2014 Kyle Fuller
        Licensed under the MIT License
        [Full MIT text here]
        """
    ),
    // Add entries for Jinja, Spectre, Stencil, Swift Collections, Swift Transformers
]
