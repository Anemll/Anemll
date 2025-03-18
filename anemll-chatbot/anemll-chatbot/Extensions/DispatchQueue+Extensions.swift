import Foundation

extension DispatchQueue {
    private static var _onceTracker = [String]()
    
    /**
     Executes a block of code only once for the lifetime of the application.
     - Parameter token: A unique token to identify this specific call site.
     - Parameter block: The code to be executed once.
     */
    static func once(token: inout Int, block: () -> Void) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        
        if token == 0 {
            token = 1
            block()
        }
    }
    
    /**
     Executes a block of code only once for the lifetime of the application.
     - Parameter token: A unique string token to identify this specific call site.
     - Parameter block: The code to be executed once.
     */
    static func once(token: String, block: () -> Void) {
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        
        if !_onceTracker.contains(token) {
            _onceTracker.append(token)
            block()
        }
    }
} 