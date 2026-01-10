ts_diff (signal : Signal<Float>, period : Int) : Signal<Float> = {
    result = subtract(x=signal, y=ts_delay(signal=signal, period=period))
}