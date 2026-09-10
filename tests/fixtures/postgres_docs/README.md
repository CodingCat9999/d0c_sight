# 픽스처 — PostgreSQL 18.6 공식 문서

`postgresql-18.6-docs.tar.gz` 에서 **원본 그대로** 복사했다. 축소하거나 손대지
않았다 — 가공하면 실물이 아니게 되고, 파서가 실제로 마주치는 구조를 놓친다.

선정 기준은 대표성이 아니라 **다양성**이다(BRIEF §6.4). 가장 크고 대표적인 문서
하나로는 버그가 숨는다. 크기·밀도·형식이 서로 다른 것을 골랐다.

| 파일 | 크기 | 왜 이것인가 |
|---|---|---|
| `runtime-config-connection.html` | 50 KB | 정의목록 지배(`<dt>` 46). `max_connections` 포함 — 예시 질문의 정답 |
| `errcodes-appendix.html` | 40 KB | 대형 단일 표(306행). SQLSTATE 코드 |
| `sql-keywords-appendix.html` | 110 KB | 코퍼스 최대 표(846행). 분할 한계 케이스 |
| `multibyte.html` | 55 KB | 표와 코드 블록 혼재 |
| `xfunc-sql.html` | 55 KB | 코드 블록 지배(`<pre>` 61, 표 0) |
| `legalnotice.html` | 2 KB | 최소 크기. 섹션 구조가 없어 인덱싱 제외 대상 |

## 출처와 라이선스

PostgreSQL 18.6 공식 문서. PostgreSQL License (BSD 계열)로 재배포 가능하다.

```
Portions Copyright (c) 1996-2025, PostgreSQL Global Development Group
Portions Copyright (c) 1994, The Regents of the University of California
```

https://www.postgresql.org/docs/18/
