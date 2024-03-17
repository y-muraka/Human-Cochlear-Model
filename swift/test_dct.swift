import Accelerate

func FFT(signal: [Float]) {

    var output = [Float](repeating: 0.0, count: signal.count)
    guard
        // Create the first setup object.
        let setupForward = vDSP_DCT_CreateSetup(nil,
                                                vDSP_Length(signal.count),
                                                .II),
        // Create the second setup object that shares resources with `setupForward`.
        let setupInverse = vDSP_DCT_CreateSetup(setupForward,
                                                vDSP_Length(signal.count),
                                                .III) else {
        NSLog("Failed to create `vDSP_DCT_CreateSetup` setup structures.")
        return
    }

    for _ in 1...800000{
                                // Perform DCT-II transform.
        vDSP_DCT_Execute(setupForward,
                        signal,
                        &output)


        // `signal` contains frequency-domain representation of the original signal.


        // Perform DCT-III transform.
        vDSP_DCT_Execute(setupInverse,
                        signal,
                        &output)
    }
    // Call `vDSP_DFT_DestroySetup` on both setup structures.
    vDSP_DFT_DestroySetup(setupForward)
    vDSP_DFT_DestroySetup(setupInverse)
    
}

let signal = [Float](repeating: 0.0, count: 512)
let start = Date()
let _ = FFT(signal: signal)
let elapsed = Date().timeIntervalSince(start)

print(elapsed)