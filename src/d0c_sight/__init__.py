"""d0c_sight — 근거를 추적할 수 있는 인프라 에러 진단 시스템.

계층과 의존 방향 (단방향, 순환 금지):

    domain  ←  sources  ←  pipeline  ←  cli

`domain` 은 계약이고 프로젝트 내부의 어떤 것도 import 하지 않는다.
이 규칙은 tests/guards/test_layering.py 가 강제한다.
"""
