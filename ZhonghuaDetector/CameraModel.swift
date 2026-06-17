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
    var previewLayer: AVCaptureVideoPreviewLayer?
}

@MainActor
final class CameraModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0
    @Published var cameraReady = false
    @Published var modelReady = false
    nonisolated(unsafe) var previewSize: CGSize = .zero
    func setPreviewLayer(_ layer: AVCaptureVideoPreviewLayer) { modelHolder.previewLayer = layer }

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
        // Xcode 26+ compiles .mlpackage → .mlmodelc at build time
        let candidates = [
            ("best", "mlmodelc"),
            ("best", "mlpackage"),
            ("best", "mlmodel"),
        ]
        var modelURL: URL?
        for (name, ext) in candidates {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                modelURL = url
                print("Found model: \(url.lastPathComponent)")
                break
            }
        }
        guard let url = modelURL else {
            print("Model not found. Bundle contents:")
            if let bundlePath = Bundle.main.resourcePath {
                for f in (try? FileManager.default.contentsOfDirectory(atPath: bundlePath)) ?? [] {
                    if f.hasPrefix("best") || f.hasPrefix("Model") { print("  \(f)") }
                }
            }
            return
        }
        do {
            let compiledURL = url.pathExtension == "mlpackage" ? try MLModel.compileModel(at: url) : url
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
        session.sessionPreset = .high

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
                conn.videoRotationAngle = 90
                conn.videoOrientation = .portrait
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

        let request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            guard let self else { return }
            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            let screenW = UIScreen.main.bounds.width
            let screenH = UIScreen.main.bounds.height
            let dets = results.compactMap { obs -> Detection? in
                guard obs.labels.first?.identifier == "zhonghua",
                      obs.confidence > 0.3 else { return nil }
                let b = obs.boundingBox
                // Vision: origin bottom-left, normalized 0-1
                // scaleFit means full image visible with letterbox
                // Camera AR (1080/1920 ≈ 0.5625) vs screen AR (393/852 ≈ 0.461)
                // scaleFit uses the smaller scale factor → camera width fills screen, top/bottom letterboxed
                let cameraAR: CGFloat = 9.0 / 16.0  // landscape sensor, portrait display: w/h = 9/16
                let screenAR = screenW / screenH
                var scale: CGFloat, offsetX: CGFloat = 0, offsetY: CGFloat = 0
                if cameraAR > screenAR {
                    // camera wider → letterbox top/bottom
                    scale = screenW
                    let dispH = screenW / cameraAR
                    offsetY = (screenH - dispH) / 2
                } else {
                    // camera taller → letterbox left/right
                    scale = screenH / cameraAR
                    offsetX = (screenW - scale) / 2
                }
                let x = offsetX + b.minX * scale
                let y = offsetY + (1 - b.maxY) * scale
                let w = b.width * scale
                let h = b.height * scale
                return Detection(boundingBox: CGRect(x: x, y: y, width: w, height: h), confidence: obs.confidence)
            }
            Task { @MainActor [weak self] in
                self?.detections = dets
            }
        }
        request.imageCropAndScaleOption = .scaleFit
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }

}
