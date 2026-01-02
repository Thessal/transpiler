ts_diff : Signal<Float> (signal : Signal<Float>, period : Int) = {
    result = subtract(x=signal, y=ts_delay(signal=signal, period=period))
}