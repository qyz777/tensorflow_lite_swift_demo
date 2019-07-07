//
//  TextClassifier.swift
//  text_swift_demo
//
//  Created by Q YiZhong on 2019/7/7.
//  Copyright © 2019 YiZhong Qi. All rights reserved.
//

import Foundation
import TensorFlowLite

fileprivate let resultCount = 3

public struct Inference {
    let confidence: Float
    let label: String
}

public typealias InferenceReslutClosure = (_ inferences: [Inference]) -> Void
public typealias CompleteClosure = () -> Void

public class TextClassifier {
    
    public static let shared = TextClassifier()
    
    private var interpreter: Interpreter!
    
    private var labels: [String] = []
    private var textIdInfo: [String: Int] = [:]
    
    /// 必须的配置是否加载完成
    public var isLoaded = false
    
    private init() {
        let options = InterpreterOptions()
        do {
            // Create the `Interpreter`.
            let modelPath = Bundle.init(for: TextClassifier.self).path(forResource: "model", ofType: "tflite")!
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
        }
    }
    
    private func loadLabels() {
        if let path = Bundle.init(for: TextClassifier.self).path(forResource: "labels", ofType: "txt") {
            let fileManager = FileManager.default
            let txtData = fileManager.contents(atPath: path)!
            let content = String.init(data: txtData, encoding: .utf8)
            let rowArray = content?.split(separator: "\n") ?? []
            for row in rowArray {
                labels.append(String(row))
            }
        }
    }
    
    private func loadTextId() {
        if let path = Bundle.init(for: TextClassifier.self).path(forResource: "text_id", ofType: "txt") {
            let fileManager = FileManager.default
            let txtData = fileManager.contents(atPath: path)!
            let content = String.init(data: txtData, encoding: .utf8)
            let rowArray = content?.split(separator: "\n") ?? []
            var i = 0
            for row in rowArray {
                textIdInfo[String(row)] = i
                i += 1
            }
        }
    }
    
    private func transformTextToId(_ text: String) -> [Int] {
        var idArray: [Int] = []
        for str in text {
            idArray.append(textIdInfo[String(str)]!)
        }
        //根据python工程中的输入设置，超出截取前面，不足后面补0
        while idArray.count < 2400 {
            idArray.append(0)
        }
        while idArray.count > 2400 {
            idArray.removeLast()
        }
        return idArray
    }
    
    
    /// 获取前N个结果
    ///
    /// - Parameter results: 预测概率值数组
    /// - Returns: 预测结果数组
    private func getTopN(results: [Float]) -> [Inference] {
        //创建元组数组 [(labelIndex: Int, confidence: Float)]
        let zippedResults = zip(labels.indices, results)
        //从大到小排序并选出前resultCount个(根据python工程中的训练，只取前10个，因为分类只有10个)
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)
        //返回前resultCount对应的标签以及预测值
        return sortedResults.map { result in Inference.init(confidence: result.1, label: labels[result.0]) }
    }
    
}

extension TextClassifier {
    
    
    /// 加载必要的配置，建议在使用前一个合适的时机调用
    /// 或者在runModel方法前调用
    ///
    /// - Parameter closure: 加载完成的回调
    public func loadInfo(_ closure: CompleteClosure? = nil) {
        guard labels.isEmpty && textIdInfo.isEmpty else {
            closure?()
            return
        }
        DispatchQueue.global().async {
            self.loadLabels()
            self.loadTextId()
            self.isLoaded = true
            closure?()
        }
    }
    
    
    /// 使用TensorFlow Lite模型进行预测
    /// 返回的结果在主线程
    ///
    /// - Parameters:
    ///   - text: 预测文本
    ///   - closure: 预测结果数组
    public func runModel(with text: String, closure: @escaping(InferenceReslutClosure)) {
        DispatchQueue.global().async {
            let idArray = self.transformTextToId(text)
            let outputTensor: Tensor
            do {
                _ = try self.interpreter.input(at: 0)
                let idData = Data.init(bytes: idArray, count: idArray.count)
                try self.interpreter.copy(idData, toInputAt: 0)
                try self.interpreter.invoke()
                outputTensor = try self.interpreter.output(at: 0)
            } catch {
                print("An error occurred while entering data: \(error.localizedDescription)")
                return
            }
            let results: [Float]
            switch outputTensor.dataType {
            case .uInt8:
                guard let quantization = outputTensor.quantizationParameters else {
                    print("No results returned because the quantization values for the output tensor are nil.")
                    return
                }
                let quantizedResults = [UInt8](outputTensor.data)
                results = quantizedResults.map {
                    quantization.scale * Float(Int($0) - quantization.zeroPoint)
                }
            case .float32:
                results = outputTensor.data.withUnsafeBytes( { (ptr: UnsafeRawBufferPointer) in
                    [Float32](UnsafeBufferPointer.init(start: ptr.baseAddress?.assumingMemoryBound(to: Float32.self), count: ptr.count))
                })
            default:
                print("Output tensor data type \(outputTensor.dataType) is unsupported for this app.")
                return
            }
            let resultArray = self.getTopN(results: results)
            DispatchQueue.main.async {
                closure(resultArray)
            }
        }
    }
    
}
