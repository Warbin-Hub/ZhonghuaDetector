import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var model = CameraModel()

    var body: some View {
        ZStack {
            GeometryReader { geo in
                CameraPreview(session: model.session, onLayer: { layer in model.setPreviewLayer(layer) })
                    .onAppear { model.previewSize = geo.size }
                    .onChange(of: geo.size) { model.previewSize = $0 }
            }.ignoresSafeArea()

            ForEach(model.detections, id: \.id) { det in
                RoundedRectangle(cornerRadius: 4)
                    .stroke(.green, lineWidth: 3)
                    .frame(width: det.boundingBox.width, height: det.boundingBox.height)
                    .position(x: det.boundingBox.midX, y: det.boundingBox.midY)
                Text(String(format: "%.0f%%", det.confidence * 100))
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.black)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(.green).cornerRadius(3)
                    .position(x: det.boundingBox.minX + 30, y: det.boundingBox.minY - 10)
            }

            VStack {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("中华烟盒检测").font(.headline.bold()).foregroundColor(.white)
                        Text(!model.cameraReady ? "启动中..." :
                             !model.modelReady ? "加载模型..." :
                             model.detections.isEmpty ? "对准烟盒正面" :
                             "检测到 \(model.detections.count) 个目标")
                            .font(.caption).foregroundColor(.white.opacity(0.7))
                    }
                    .padding(12).background(.ultraThinMaterial).cornerRadius(10)
                    .padding(.top, 60).padding(.leading, 16)
                    Spacer()
                }
                Spacer()
                HStack(spacing: 12) {
                    Text("\(Int(model.fps)) FPS").font(.caption.monospaced())
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(.ultraThinMaterial).cornerRadius(6)
                    Text("\(model.detections.count) 目标").font(.caption.monospaced())
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(.ultraThinMaterial).cornerRadius(6)
                }
                .foregroundColor(.white).padding(.bottom, 40)
            }
        }
        .task { await model.start() }
        .onDisappear { model.stop() }
    }
}

class PreviewVC: UIViewController {
    let session: AVCaptureSession
    var onLayer: ((AVCaptureVideoPreviewLayer) -> Void)?
    var previewLayer: AVCaptureVideoPreviewLayer!
    init(session: AVCaptureSession) { self.session = session; super.init(nibName: nil, bundle: nil) }
    required init?(coder: NSCoder) { fatalError() }
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspect
        view.layer.addSublayer(previewLayer)
        onLayer?(previewLayer)
    }
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }
}

struct CameraPreview: UIViewControllerRepresentable {
    let session: AVCaptureSession
    var onLayer: ((AVCaptureVideoPreviewLayer) -> Void)?
    func makeUIViewController(context: Context) -> PreviewVC {
        let vc = PreviewVC(session: session)
        vc.onLayer = onLayer
        return vc
    }
    func updateUIViewController(_ vc: PreviewVC, context: Context) {}
}
