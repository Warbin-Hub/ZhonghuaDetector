import AVFoundation
import Vision
import UIKit

struct Detection: Identifiable {
    let id = UUID()
    let boundingBox: CGRect
    let confidence: Float
}

final class ModelHolder: @unchecked Sendable {
    var model: VNCoreMLModel?
}

@MainActor
final class CameraModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0
    @Published var cameraReady = false
    @Published var modelReady = false

    private let modelHolder = ModelHolder()
    private var fpsCounter: Int = 0
    private var fpsTimer: Timer?

    func start() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .notDetermined {
            guard await AVCaptureDevice.requestAccess(for: .video) else {
                print("Camera permission denied")
                return
            }
        } else if status != .authorized {
            print("Camera not authorized: \(status)")
            return
        }
        loadModel()
        setupCamera()
    }

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "best", withExtension: "mlpackage") else {
            print("Model file not found in bundle")
            return
        }
        do {
            let compiledURL = try MLModel.compileModel(at: url)
            let mlModel = try MLModel(contentsOf: compiledURL)
            modelHolder.model = try VNCoreMLModel(for: mlModel)
            modelReady = true
            print("Model loaded OK")
        } catch {
            print("Model load error: \(error)")
        }
    }

    private func setupCamera() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("No back camera found")
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .vga640x480

        guard let input = try? AVCaptureDeviceInput(device: device) else {
            print("Failed to create camera input")
            session.commitConfiguration()
            return
        }
        guard session.canAddInput(input) else {
            print("Cannot add camera input")
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "inference", qos: .userInteractive))

        guard session.canAddOutput(output) else {
            print("Cannot add video output")
            session.commitConfiguration()
            return
        }
        session.addOutput(output)

        if let conn = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                conn.videoRotationAngle = 0
            }
        }

        session.commitConfiguration()
        print("Camera configured, starting session...")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
            Task { @MainActor [weak self] in
                self?.cameraReady = true
                print("Camera session running")
            }
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
        let screenW = UIScreen.main.bounds.width
        let screenH = UIScreen.main.bounds.height
        let scale = max(screenW / rect.height, screenH / rect.width)
        let viewW = rect.height * scale
        let viewH = rect.width * scale
        return CGRect(
            x: (screenW - viewW) / 2 + (1 - rect.maxY) * viewW,
            y: (screenH - viewH) / 2 + rect.minX * viewH,
            width: viewW,
            height: viewH
        )
    }
}
