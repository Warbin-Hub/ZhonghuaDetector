import AVFoundation
import Vision

struct Detection: Identifiable {
    let id = UUID()
    let boundingBox: CGRect
    let confidence: Float
}

/// Thread-safe holder for VNCoreMLModel (which is already thread-safe per Apple docs)
final class ModelHolder: @unchecked Sendable {
    var model: VNCoreMLModel?
}

@MainActor
final class CameraModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0

    private let modelHolder = ModelHolder()
    private var fpsCounter: Int = 0
    private var fpsTimer: Timer?

    func start() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status != .authorized {
            guard await AVCaptureDevice.requestAccess(for: .video) else { return }
        }
        loadModel()
        setupCamera()
    }

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "best", withExtension: "mlpackage") else {
            print("best.mlpackage not in bundle")
            return
        }
        guard let compiledURL = try? MLModel.compileModel(at: url),
              let mlModel = try? MLModel(contentsOf: compiledURL),
              let vnModel = try? VNCoreMLModel(for: mlModel) else {
            print("Model load failed")
            return
        }
        modelHolder.model = vnModel
        print("Model ready")
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1920x1080

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration(); return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "inference", qos: .userInteractive))
        guard session.canAddOutput(output) else { session.commitConfiguration(); return }
        session.addOutput(output)

        if let conn = output.connection(with: .video) {
            conn.videoRotationAngle = 90
            conn.isVideoMirrored = false
        }

        session.commitConfiguration()
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }

        fpsTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.fps = Double(self.fpsCounter)
                self.fpsCounter = 0
            }
        }
    }

    func stop() {
        session.stopRunning()
        fpsTimer?.invalidate()
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let vnModel = modelHolder.model else { return }

        Task { @MainActor in self.fpsCounter += 1 }

        let request = VNCoreMLRequest(model: vnModel) { req, _ in
            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            let dets = results.compactMap { obs -> Detection? in
                guard obs.labels.first?.identifier == "zhonghua",
                      obs.confidence > 0.5 else { return nil }
                return Detection(boundingBox: obs.boundingBox, confidence: obs.confidence)
            }
            Task { @MainActor [weak self] in
                self?.detections = dets
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }

    func visionToScreen(_ rect: CGRect) -> CGRect {
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height
        let scale = max(w / rect.height, h / rect.width)
        let vw = rect.height * scale
        let vh = rect.width * scale
        return CGRect(
            x: (w - vw) / 2 + (1 - rect.maxY) * vw,
            y: (h - vh) / 2 + rect.minX * vh,
            width: vw,
            height: vh
        )
    }
}
