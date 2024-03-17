import Accelerate

func MV(signal: [Float]){

    let dim = Int32(signal.count)
    var output = [Float](repeating: 0.0, count: signal.count)
    for _ in 1...800000{ 
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0, signal, dim, signal, 1, 0, &output, 1);
    }
}

let signal = [Float](repeating: 0.0, count: 512)
let start = Date()
let _ = MV(signal: signal)
let elapsed = Date().timeIntervalSince(start)

print(elapsed)