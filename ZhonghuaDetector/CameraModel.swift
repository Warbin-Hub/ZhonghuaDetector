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
        session.sessionPreset = .hd1280x720

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

        let frameW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            guard let self else { return }
            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            let screenW = UIScreen.main.bounds.width
            let screenH = UIScreen.main.bounds.height
            let cameraAR = frameW / frameH // actual camera aspect ratio
            let screenAR = screenW / screenH
            // scaleFit: use smaller scale factor to fit full image
            var scale: CGFloat, ox: CGFloat = 0, oy: CGFloat = 0
            if cameraAR > screenAR {
                scale = screenW
                oy = (screenH - screenW / cameraAR) / 2
            } else {
                scale = screenH * cameraAR
                ox = (screenW - scale) / 2
            }
            let dets = results.compactMap { obs -> Detection? in
                guard obs.labels.first?.identifier == "zhonghua",
                      obs.confidence > 0.3 else { return nil }
                let b = obs.boundingBox
                // Vision output is in RAW frame coords (1280w×720h landscape)
                // Display is rotated 90° → 720w×1280h portrait
                // Swap X↔Y: Vision width → display height, Vision height → display width
                let screenX = ox + (1 - b.maxY) * scale
                let screenY = oy + b.minX * scale
                let screenW = b.height * scale
                let screenH = b.width * scale
                return Detection(boundingBox: CGRect(x: screenX, y: screenY, width: screenW, height: screenH), confidence: obs.confidence)
            }
            Task { @MainActor [weak self] in
                self?.detections = dets
            }
        }
        request.imageCropAndScaleOption = .scaleFit
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }

}
