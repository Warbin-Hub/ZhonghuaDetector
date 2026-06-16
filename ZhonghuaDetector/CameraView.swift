import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var model = CameraModel()

    var body: some View {
        ZStack {
            CameraPreview(session: model.session)
                .ignoresSafeArea()

            ForEach(model.detections, id: \.id) { det in
                let rect = model.visionToScreen(det.boundingBox)
                RoundedRectangle(cornerRadius: 4)
                    .stroke(.yellow, lineWidth: 2)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)

                Text(String(format: "%.0f%%", det.confidence * 100))
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.black)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.yellow)
                    .cornerRadius(3)
                    .position(x: rect.minX + 28, y: rect.minY - 10)
            }

            VStack {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("中华烟盒检测")
                            .font(.headline.bold())
                            .foregroundColor(.white)
                        Text(model.detections.isEmpty
                             ? "对准烟盒正面"
                             : "检测到 \(model.detections.count) 个目标")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.7))
                    }
                    .padding(12)
                    .background(.ultraThinMaterial)
                    .cornerRadius(10)
                    .padding(.top, 60)
                    .padding(.leading, 16)
                    Spacer()
                }
                Spacer()

                HStack(spacing: 12) {
                    Text("\(Int(model.fps)) FPS")
                        .font(.caption.monospaced())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial)
                        .cornerRadius(6)
                    Text("\(model.detections.count) 目标")
                        .font(.caption.monospaced())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial)
                        .cornerRadius(6)
                }
                .foregroundColor(.white)
                .padding(.bottom, 40)
            }
        }
        .task { await model.start() }
        .onDisappear { model.stop() }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        view.layer.addSublayer(preview)
        context.coordinator.preview = preview
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.preview?.frame = uiView.bounds
    }
    func makeCoordinator() -> Coordinator { Coordinator() }
    class Coordinator { var preview: AVCaptureVideoPreviewLayer? }
}
