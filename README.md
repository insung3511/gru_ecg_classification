# ECG Classification by GRU Model
- Reference paper : [Interpretation of Electrocardiogram Heartbeat by CNN and GRU](https://www.hindawi.com/journals/cmmm/2021/6534942/)

이전에 진행 했던 [ECG Classification by GB-DBN Model](https://github.com/insung3511) 의 연장선이다. 알아야 할 점은 이번에도 논문을 참고로 한다는 점이다. 어디까지나 참고이므로 논문 내용 그대로 가진 않을수도 있다. 실제 Pre-processing 과정에서 차이가 발생 할 수도 있고, 혹은 다른 문제에 봉착 할 수도 있기 때문이다. 암튼 그게 중요한게 아니고 모델의 구조는 위 논문을 참고하길 바란다. 개인적으로는 위 논문 정말 모델을 그대로 구현하기에는 괜찮다고 생각된다.

이번에도 Pytorch 로 진행될 예정이다. Data는 이전에 Pre-processing을 끝낸 [ecg-rr](https://github.com/ecg-rr) 에서 활용했던 데이터를 갖고와 진행 할 것이다.

이 외의 질문이나 혹은 문제점이 있다면 이메일 혹은 Issue를 해준다면 언제든지 환영이다. :)

