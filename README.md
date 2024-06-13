# Number Recognizer

This SwiftUI application recognizes handwritten digits from 1 to 10 using a neural network. The neural network model is trained using the MNIST dataset and integrated into the SwiftUI app using Core ML and Vision frameworks.

## Features
- Draw digits on the canvas.
- Predict the digit using a pre-trained neural network model.
- Display the recognized digit.

## Requirements
- Xcode 12.0 or later
- iOS 14.0 or later

## Setup Instructions

### 1. Model Training and Conversion
1. Ensure you have Python and the required libraries installed. You can install the libraries using pip:
    ```sh
    pip install tensorflow coremltools
    ```

2. Use the following Python script to train the model and convert it to Core ML format:
    ```python
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    import coremltools as ct

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\ntest accuracy:', test_acc)

    model.save('mnist_model.h5')
    mlmodel = ct.convert('mnist_model.h5', inputs=[ct.ImageType(name="image", shape=(1, 28, 28, 1), scale=1/255.0)])
    mlmodel.save('MNISTClassifier.mlmodel')
    ```

3. After running the script, you will have a `MNISTClassifier.mlmodel` file.

### 2. Integrate the Model into Xcode
1. Open your Xcode project.
2. Drag and drop the `MNISTClassifier.mlmodel` file into your Xcode project navigator.

### 3. SwiftUI Implementation
Use the following SwiftUI code for the application:

```swift
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
```

## Running the App
1. Open the project in Xcode.
2. Build and run the app on a simulator or a physical device.
3. Draw a digit (1-10) on the canvas and press the "Recognize" button to see the predicted digit.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
