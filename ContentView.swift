import SwiftUI
import CoreML
import Vision

struct ContentView: View {
    @State private var drawnImage: UIImage?
    @State private var recognizedNumber: String = "?"

    var body: some View {
        VStack {
            Text("Draw a number")
                .font(.largeTitle)
                .padding()

            DrawingCanvas(image: $drawnImage)
                .frame(width: 300, height: 300)
                .border(Color.black, width: 1)

            Button(action: recognizeNumber) {
                Text("Recognize")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()

            Text("Recognized Number: \(recognizedNumber)")
                .font(.title)
                .padding()
        }
    }

    private func recognizeNumber() {
        guard let drawnImage = drawnImage else { return }

        let model = try! VNCoreMLModel(for: MNISTClassifier().model)
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let bestResult = results.first else {
                return
            }
            DispatchQueue.main.async {
                recognizedNumber = bestResult.identifier
            }
        }

        let handler = VNImageRequestHandler(cgImage: drawnImage.cgImage!, options: [:])
        try? handler.perform([request])
    }
}

struct DrawingCanvas: UIViewRepresentable {
    @Binding var image: UIImage?

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let canvasView = CanvasView()
        canvasView.backgroundColor = .white
        canvasView.delegate = context.coordinator
        view.addSubview(canvasView)

        canvasView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            canvasView.topAnchor.constraint(equalTo: view.topAnchor),
            canvasView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            canvasView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            canvasView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])

        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) { }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, CanvasViewDelegate {
        var parent: DrawingCanvas

        init(_ parent: DrawingCanvas) {
            self.parent = parent
        }

        func didEndDrawing(_ canvasView: CanvasView, image: UIImage) {
            parent.image = image
        }
    }
}

protocol CanvasViewDelegate: AnyObject {
    func didEndDrawing(_ canvasView: CanvasView, image: UIImage)
}

class CanvasView: UIView {
    weak var delegate: CanvasViewDelegate?

    private var lines: [Line] = []
    private let path = UIBezierPath()

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let newLine = Line(points: [touch.location(in: self)])
        lines.append(newLine)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let currentPoint = touch.location(in: self)
        guard var lastLine = lines.popLast() else { return }
        lastLine.points.append(currentPoint)
        lines.append(lastLine)
        setNeedsDisplay()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        UIGraphicsBeginImageContext(bounds.size)
        drawHierarchy(in: bounds, afterScreenUpdates: true)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        delegate?.didEndDrawing(self, image: image!)
    }

    override func draw(_ rect: CGRect) {
        UIColor.black.setStroke()
        path.lineWidth = 3
        path.lineCapStyle = .round
        lines.forEach { line in
            guard let firstPoint = line.points.first else { return }
            path.move(to: firstPoint)
            for point in line.points.dropFirst() {
                path.addLine(to: point)
            }
        }
        path.stroke()
    }

    struct Line {
        var points: [CGPoint]
    }
}

@main
struct NumberRecognizerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
