import SwiftUI

@main
struct ZhonghuaDetectorApp: App {
    var body: some Scene {
        WindowGroup {
            CameraView()
                .ignoresSafeArea()
                .preferredColorScheme(.dark)
        }
    }
}
