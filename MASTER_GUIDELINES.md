# 마스터 개발 가이드라인 v3.0

## "동작하는 코드"가 아닌 "프로덕션에서 안정적이고 최적화된 코드"를 생성한다

> **근거**: CodeRabbit 2025 보고서에 따르면 AI 생성 코드는 인간 코드 대비 1.7배 더 많은 결함을 포함한다.
> 로직/정확성 오류 1.75배, 보안 취약점 1.57배, 성능 문제 1.42배, 과도한 I/O 8배 더 빈번하다.
> 이 가이드라인은 해당 통계적 실패 패턴의 90% 이상을 사전 차단하기 위해 설계되었다.

---

## Part 0: 구현 전 필수 분석 (절대 생략 불가)

코드를 한 줄이라도 작성하기 **전에** 반드시 수행한다.
이 단계를 생략한 코드는 전부 삭제하고 처음부터 다시 작성한다.

### 0.1 의존성 맵 생성

작업 대상 파일에서 다음을 **전부 Read로 스캔**하고 목록을 작성한다:

| 언어 | 스캔 대상 |
|------|---------|
| Java/JSP | `import`, `extends`, `implements`, `@Autowired`, `@Inject`, XML 설정 |
| C/C++ | `#include`, 전방 선언, `CMakeLists.txt`, 링크 라이브러리 |
| JavaScript | `import`, `require`, `package.json` dependencies |
| CSS | `@import`, 상위 셀렉터, 미디어 쿼리 의존 |
| SQL | 참조하는 테이블, FK 관계, 인덱스, 트리거 |

### 0.2 연관 파일 확인

위에서 파악한 의존성 중 수정에 영향받는 파일들을 Read로 열어서:
- 실제 메서드/함수 시그니처 확인
- 실제 변수명/상수명/컬럼명 확인
- 반환 타입, 파라미터 순서, null 가능성 확인

### 0.3 구현 전 자가 질문 (5가지 모두 답한 후 시작)

| # | 질문 | 확인 항목 |
|---|------|---------|
| Q1 | 이미 구현된 것을 중복 구현하는가? | 기존 유틸, 캐시, 헬퍼를 검색했는가? |
| Q2 | 데이터 규모는 얼마인가? | 1건? 100건? 10만건? 규모에 따라 전략이 달라진다 |
| Q3 | 이 코드는 어느 스레드/컨텍스트에서 실행되는가? | 메인 스레드? 워커? 비동기? 동기화 필요 여부 |
| Q4 | 호출 빈도는 얼마인가? | 한 번? 루프? 초당 수백 회? |
| Q5 | 실패 시 어떻게 복구하는가? | 예외 처리, 롤백, 재시도 전략 |

### 0.4 분석 결과 보고 (코드 작성 전 필수 출력)

```
[분석 완료 보고]
- 대상 파일: xxx
- 확인한 의존성: A, B, C
- 사용할 기존 메서드: methodA(String, int), funcB(const char*)
- 영향받는 파일: D (A를 상속), E (A를 호출)
- 잠재적 충돌: 없음 / 있음(상세)
- 데이터 규모: N건
- 호출 빈도: X회/초
```

이 보고 없이 코드 작성을 시작하지 않는다.

---

## Part 1: 추측 코딩 완전 금지 (Zero Assumption)

> **통계**: AI 로직/정확성 오류의 주요 원인은 존재하지 않는 API 호출, 잘못된 시그니처 사용이다.

### 1.1 존재 확인 없이 사용 금지

```java
// ❌ Java - 추측
user.getFullName();        // getFullName()이 있을 것이라 가정 → 컴파일 에러

// ✅ Java - 확인 후 사용
// User.java 확인 → getName() + " " + getLastName() 으로 구성해야 함
user.getName() + " " + user.getLastName();
```

```cpp
// ❌ C++ - 추측
encoder->flush();          // flush() 메서드가 있을 것이라 가정

// ✅ C++ - 헤더 확인 후 사용
// encoder.h 확인 → drain()이 정의되어 있음
encoder->drain();
```

```javascript
// ❌ JavaScript - 추측
response.data.users.map(u => u.email);  // 구조를 가정

// ✅ JavaScript - 방어적 접근
const users = response?.data?.users ?? [];
users.map(u => u.email ?? '');
```

### 1.2 라이브러리/프레임워크 버전 확인

- 사용할 API가 현재 프로젝트의 라이브러리 버전에서 지원하는지 확인
- `pom.xml`, `package.json`, `CMakeLists.txt` 에서 버전 확인 후 사용
- deprecated된 API를 사용하지 않기

### 1.3 플랫폼/환경 분기 확인 (C/C++)

```cpp
// ❌ 플랫폼 무시
auto handle = CreateFileW(...);   // Linux에서 컴파일 실패

// ✅ 플랫폼 분기
#ifdef _WIN32
  auto handle = CreateFileW(...);
#elif __linux__
  int fd = open(...);
#endif
```

---

## Part 2: 기존 자산 활용 강제 (중복 구현 금지)

> **통계**: AI 코드 복제(cloning)가 4배 증가. 기존 코드 재사용보다 복붙이 더 빈번.

### 2.1 검색 우선, 구현 나중

새로운 유틸리티, 헬퍼, 공통 함수를 작성하기 전에 **반드시** 기존 코드를 검색한다:

```bash
# 프로젝트 내 유사 기능 검색
grep -r "키워드" src/
grep -r "functionName" --include="*.java" .
grep -r "className" --include="*.js" .
grep -r "함수명" --include="*.cpp" src/
```

### 2.2 캐시가 있으면 캐시를 사용

```java
// ❌ 캐시를 만들어놓고 매번 DB 조회
public UserInfo getUser(long userId) {
    return userRepository.findById(userId);      // 매번 DB 호출
}

// ✅ 캐시 우선 조회
public UserInfo getUser(long userId) {
    UserInfo cached = userCache.get(userId);      // 캐시 먼저
    if (cached != null) return cached;
    UserInfo user = userRepository.findById(userId);
    userCache.put(userId, user);
    return user;
}
```

### 2.3 기존 패턴/컨벤션 준수

- 기존 네이밍 컨벤션을 그대로 따른다 (camelCase, snake_case 등)
- 기존 디자인 패턴을 그대로 따른다 (Singleton, Factory 등)
- 기존 로깅 방식을 그대로 따른다 (log4j, BOOST_LOG, console.log 등)
- 새로운 패턴 도입은 반드시 사전 협의

---

## Part 3: 데이터베이스 최적화 (MySQL / MariaDB)

> **통계**: AI의 과도한 I/O 발생률이 인간 대비 8배. DB 쿼리가 가장 큰 비중.

### 3.1 쿼리 작성 원칙

```sql
-- ❌ 전체 조회
SELECT * FROM users;

-- ✅ 필요한 컬럼만, 조건과 제한 포함
SELECT user_id, name, email
FROM users
WHERE status = 'ACTIVE'
LIMIT 100;
```

### 3.2 N+1 문제 근절

```java
// ❌ N+1: 루프 안에서 DB 호출
for (Order order : orders) {
    User user = userRepository.findById(order.getUserId());  // N번 쿼리
}

// ✅ IN 절로 한 번에 조회
List<Long> userIds = orders.stream().map(Order::getUserId).distinct().toList();
Map<Long, User> userMap = userRepository.findByIdIn(userIds)
    .stream().collect(Collectors.toMap(User::getId, Function.identity()));
for (Order order : orders) {
    User user = userMap.get(order.getUserId());               // 메모리 조회
}
```

### 3.3 인덱스 활용

```sql
-- ❌ 인덱스 무효화
SELECT * FROM orders WHERE YEAR(created_at) = 2026;           -- 함수 사용으로 인덱스 무효화
SELECT * FROM users WHERE name LIKE '%검색어%';               -- 앞쪽 와일드카드

-- ✅ 인덱스 활용
SELECT * FROM orders WHERE created_at >= '2026-01-01' AND created_at < '2027-01-01';
SELECT * FROM users WHERE name LIKE '검색어%';                -- 뒤쪽 와일드카드만
```

### 3.4 트랜잭션 범위 최소화

```java
// ❌ 트랜잭션 안에서 외부 API 호출
@Transactional
public void processOrder(Order order) {
    orderRepository.save(order);
    externalPaymentApi.charge(order);    // 외부 API 실패 시 DB 커넥션 장시간 점유
    emailService.sendConfirmation(order); // 이메일 전송까지 트랜잭션 범위
}

// ✅ 트랜잭션 범위 최소화
@Transactional
public void saveOrder(Order order) {
    orderRepository.save(order);          // DB 작업만 트랜잭션
}

public void processOrder(Order order) {
    saveOrder(order);                     // 트랜잭션 종료
    externalPaymentApi.charge(order);     // 트랜잭션 밖
    emailService.sendConfirmation(order); // 트랜잭션 밖
}
```

### 3.5 SQL 인젝션 방지

```java
// ❌ 문자열 결합 (SQL Injection 취약)
String sql = "SELECT * FROM users WHERE name = '" + name + "'";

// ✅ PreparedStatement / 파라미터 바인딩
String sql = "SELECT * FROM users WHERE name = ?";
PreparedStatement ps = conn.prepareStatement(sql);
ps.setString(1, name);
```

```javascript
// ❌ 문자열 템플릿 (SQL Injection 취약)
const sql = `SELECT * FROM users WHERE id = ${userId}`;

// ✅ 파라미터 바인딩
const sql = 'SELECT * FROM users WHERE id = ?';
connection.query(sql, [userId]);
```

### 3.6 대량 데이터 처리

```java
// ❌ 10만 건을 한 번에 메모리 로드
List<User> allUsers = userRepository.findAll();

// ✅ 페이징 처리
int page = 0, size = 1000;
Page<User> users;
do {
    users = userRepository.findAll(PageRequest.of(page++, size));
    processBatch(users.getContent());
} while (users.hasNext());
```

```sql
-- ❌ 대량 INSERT를 한 건씩
INSERT INTO logs (msg) VALUES ('a');
INSERT INTO logs (msg) VALUES ('b');
-- ... 1만번 반복

-- ✅ 배치 INSERT
INSERT INTO logs (msg) VALUES ('a'), ('b'), ('c'), ... ;
```

---

## Part 4: 메모리 관리

### 4.1 Java / JSP

```java
// ❌ 루프 안 불필요한 객체 생성
for (int i = 0; i < 100000; i++) {
    String result = new StringBuilder().append("item_").append(i).toString();
}

// ✅ 루프 밖에서 재사용
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 100000; i++) {
    sb.setLength(0);
    sb.append("item_").append(i);
    String result = sb.toString();
}
```

```java
// ❌ 리소스 미해제
InputStream is = new FileInputStream("file.txt");
// ... 예외 발생 시 is가 해제되지 않음

// ✅ try-with-resources 필수
try (InputStream is = new FileInputStream("file.txt")) {
    // ... 자동 해제 보장
}
```

```java
// ❌ 크기 미지정 컬렉션
List<Item> items = new ArrayList<>();           // 내부 배열 반복 확장

// ✅ 예상 크기 지정
List<Item> items = new ArrayList<>(expectedSize);
Map<String, Object> map = new HashMap<>(expectedSize * 4 / 3 + 1);
```

### 4.2 C / C++

```cpp
// ❌ Raw 포인터 직접 관리
char* buffer = new char[1024];
process(buffer);
// delete[] buffer;   ← 예외 발생 시 누수

// ✅ 스마트 포인터 / RAII
auto buffer = std::make_unique<char[]>(1024);
process(buffer.get());
// 자동 해제
```

```cpp
// ❌ 댕글링 포인터
int* getData() {
    int local = 42;
    return &local;          // 함수 종료 후 무효한 포인터
}

// ✅ 값 반환 또는 힙 할당
int getData() {
    return 42;              // 값 복사
}
std::unique_ptr<int> getData() {
    return std::make_unique<int>(42);   // 소유권 이전
}
```

```cpp
// ❌ 버퍼 오버플로우
char dest[10];
strcpy(dest, userInput);     // 길이 검사 없음

// ✅ 크기 제한 복사
char dest[10];
strncpy(dest, userInput, sizeof(dest) - 1);
dest[sizeof(dest) - 1] = '\0';
// 또는 std::string 사용
```

### 4.3 JavaScript

```javascript
// ❌ 클로저로 인한 메모리 누수
function createHandler() {
    const hugeData = new Array(1000000).fill('x');
    return function() {
        console.log(hugeData.length);   // hugeData가 GC되지 않음
    };
}

// ✅ 필요한 값만 캡처
function createHandler() {
    const hugeData = new Array(1000000).fill('x');
    const len = hugeData.length;        // 필요한 값만 추출
    return function() {
        console.log(len);              // hugeData는 GC 가능
    };
}
```

```javascript
// ❌ 이벤트 리스너 미해제
element.addEventListener('click', handler);
// 컴포넌트 제거 시 handler가 계속 참조

// ✅ 정리(cleanup) 보장
element.addEventListener('click', handler);
// 제거 시
element.removeEventListener('click', handler);

// React: useEffect cleanup
useEffect(() => {
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
}, []);
```

---

## Part 5: Null 안전성 / 방어적 프로그래밍

> **통계**: AI 코드는 null 체크, 조기 반환, 가드 로직을 자주 누락한다.

### 5.1 Java

```java
// ❌ NullPointerException 위험
String name = user.getAddress().getCity().getName();

// ✅ 단계별 null 체크
if (user == null || user.getAddress() == null) return defaultValue;
City city = user.getAddress().getCity();
String name = (city != null) ? city.getName() : "Unknown";

// ✅ Optional 활용 (Java 8+)
String name = Optional.ofNullable(user)
    .map(User::getAddress)
    .map(Address::getCity)
    .map(City::getName)
    .orElse("Unknown");
```

### 5.2 JavaScript

```javascript
// ❌ TypeError: Cannot read property
const city = response.data.user.address.city;

// ✅ Optional Chaining + Nullish Coalescing
const city = response?.data?.user?.address?.city ?? 'Unknown';
```

### 5.3 C/C++

```cpp
// ❌ nullptr 역참조
void process(Config* config) {
    std::string name = config->getName();  // config이 nullptr이면 크래시
}

// ✅ 가드 절 (Guard Clause)
void process(Config* config) {
    if (!config) {
        LOG_WARN("config is null");
        return;
    }
    std::string name = config->getName();
}
```

### 5.4 SQL 결과 처리

```java
// ❌ 쿼리 결과를 무조건 신뢰
User user = userRepository.findByEmail(email);
String name = user.getName();               // user가 null이면 NPE

// ✅ 존재 여부 확인
User user = userRepository.findByEmail(email);
if (user == null) {
    throw new UserNotFoundException("User not found: " + email);
}
```

---

## Part 6: 동시성 / 스레드 안전성

> **통계**: AI의 동시성 프리미티브 오용, 잘못된 의존성 순서가 빈번하게 발생.

### 6.1 Java

```java
// ❌ 비 thread-safe 싱글톤
public class Cache {
    private static Cache instance;
    public static Cache getInstance() {
        if (instance == null) instance = new Cache();   // Race condition
        return instance;
    }
}

// ✅ Thread-safe 싱글톤 (Holder 패턴)
public class Cache {
    private Cache() {}
    private static class Holder {
        static final Cache INSTANCE = new Cache();
    }
    public static Cache getInstance() {
        return Holder.INSTANCE;
    }
}
```

```java
// ❌ ConcurrentModificationException
for (User user : userList) {
    if (user.isExpired()) userList.remove(user);
}

// ✅ Iterator 또는 removeIf 사용
userList.removeIf(User::isExpired);

// ✅ 동시 접근 시 ConcurrentHashMap
Map<String, Session> sessions = new ConcurrentHashMap<>();
```

### 6.2 C/C++

```cpp
// ❌ 뮤텍스 없이 공유 자원 접근
int frameCount = 0;
void captureThread() { frameCount++; }       // Data race
void statsThread()   { log(frameCount); }    // Data race

// ✅ atomic 또는 mutex
std::atomic<int> frameCount{0};
void captureThread() { frameCount.fetch_add(1); }
void statsThread()   { log(frameCount.load()); }
```

```cpp
// ❌ 데드락 위험 (락 순서 불일치)
void threadA() { lock(mutexA); lock(mutexB); }
void threadB() { lock(mutexB); lock(mutexA); }   // 데드락!

// ✅ 항상 동일한 순서로 락 / std::scoped_lock 사용
void threadA() { std::scoped_lock lk(mutexA, mutexB); }
void threadB() { std::scoped_lock lk(mutexA, mutexB); }
```

### 6.3 JavaScript (비동기)

```javascript
// ❌ Promise 에러 무시
fetch('/api/data').then(res => res.json()).then(process);

// ✅ 에러 핸들링 필수
try {
    const res = await fetch('/api/data');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    process(data);
} catch (err) {
    console.error('API call failed:', err);
    showErrorToUser(err.message);
}
```

```javascript
// ❌ 루프 내 순차 await (N+1과 유사)
for (const id of ids) {
    const data = await fetch(`/api/items/${id}`);   // 하나씩 순차 호출
}

// ✅ 병렬 실행
const results = await Promise.all(
    ids.map(id => fetch(`/api/items/${id}`).then(r => r.json()))
);
```

---

## Part 7: 보안 (Security)

> **통계**: AI 코드의 보안 취약점 발생률 1.57배. 비밀번호 처리 오류 1.88배, 안전하지 않은 객체 참조 1.91배.

### 7.1 인증/인가

```java
// ❌ 비밀번호 평문 저장
user.setPassword(rawPassword);

// ✅ 해시 저장
user.setPassword(BCrypt.hashpw(rawPassword, BCrypt.gensalt()));
```

```java
// ❌ 하드코딩된 시크릿
String apiKey = "sk-12345abcde";

// ✅ 환경 변수 / 설정 파일
String apiKey = System.getenv("API_KEY");
```

### 7.2 입력 검증

모든 사용자 입력은 **악의적**이라고 가정한다:

```java
// ❌ 사용자 입력을 그대로 사용
String filename = request.getParameter("file");
File f = new File("/uploads/" + filename);        // Path Traversal: ../../etc/passwd

// ✅ 입력 검증 및 정규화
String filename = request.getParameter("file");
if (filename.contains("..") || filename.contains("/") || filename.contains("\\")) {
    throw new SecurityException("Invalid filename");
}
Path path = Paths.get("/uploads", filename).normalize();
if (!path.startsWith("/uploads")) {
    throw new SecurityException("Path traversal detected");
}
```

### 7.3 XSS 방지 (JSP / JavaScript)

```jsp
<%-- ❌ 미이스케이프 출력 --%>
<p>${userInput}</p>

<%-- ✅ JSTL 이스케이프 --%>
<p><c:out value="${userInput}" /></p>
<%-- 또는 --%>
<p>${fn:escapeXml(userInput)}</p>
```

```javascript
// ❌ innerHTML로 사용자 입력 삽입
element.innerHTML = userInput;           // XSS 취약

// ✅ textContent 사용
element.textContent = userInput;         // HTML로 해석되지 않음
```

### 7.4 CSRF 방지

```html
<!-- ❌ 토큰 없는 폼 -->
<form method="POST" action="/transfer">
    <input name="amount" value="1000">
</form>

<!-- ✅ CSRF 토큰 포함 -->
<form method="POST" action="/transfer">
    <input type="hidden" name="_csrf" value="${_csrf.token}">
    <input name="amount" value="1000">
</form>
```

---

## Part 8: 성능 최적화

> **통계**: AI의 과도한 I/O 발생률이 인간 대비 8배.

### 8.1 I/O 최소화

```java
// ❌ 루프 내 파일/네트워크 호출
for (String id : ids) {
    String data = httpClient.get("/api/item/" + id);   // N번 HTTP 호출
}

// ✅ 배치 API 호출
String data = httpClient.post("/api/items/batch", ids);  // 1번 호출
```

### 8.2 알고리즘 복잡도

```java
// ❌ O(n²) — 대량 데이터에서 치명적
for (User u1 : users) {
    for (User u2 : users) {
        if (u1.getId().equals(u2.getFriendId())) { ... }
    }
}

// ✅ O(n) — Map 활용
Map<Long, User> userMap = users.stream()
    .collect(Collectors.toMap(User::getId, u -> u));
for (User u : users) {
    User friend = userMap.get(u.getFriendId());   // O(1) 조회
}
```

### 8.3 문자열 처리

```java
// ❌ 루프 내 문자열 결합 (매번 새 String 객체 생성)
String result = "";
for (String s : list) {
    result += s + ",";
}

// ✅ StringBuilder 사용
StringBuilder sb = new StringBuilder(list.size() * 20);
for (String s : list) {
    if (sb.length() > 0) sb.append(',');
    sb.append(s);
}
String result = sb.toString();

// ✅ 더 나은 방법: String.join
String result = String.join(",", list);
```

### 8.4 CSS 성능

```css
/* ❌ 비효율적 셀렉터 (브라우저는 오른쪽에서 왼쪽으로 매칭) */
div > ul > li > a.link { }
* { box-sizing: border-box; }      /* 전체 선택자 남용 */

/* ✅ 클래스 기반 셀렉터 */
.nav-link { }
```

```css
/* ❌ 과도한 리플로우 유발 */
.box {
    width: 100px;
    /* JS에서 매 프레임 offsetWidth 읽기 → 강제 리플로우 */
}

/* ✅ transform 사용 (GPU 가속, 리플로우 없음) */
.box {
    transform: translateX(100px);
    will-change: transform;
}
```

### 8.5 JavaScript 성능

```javascript
// ❌ DOM 반복 접근
for (let i = 0; i < 1000; i++) {
    document.getElementById('list').innerHTML += `<li>${i}</li>`;  // 1000번 리플로우
}

// ✅ DocumentFragment 또는 한 번에 삽입
const fragment = document.createDocumentFragment();
for (let i = 0; i < 1000; i++) {
    const li = document.createElement('li');
    li.textContent = i;
    fragment.appendChild(li);
}
document.getElementById('list').appendChild(fragment);   // 1번 리플로우
```

### 8.6 C/C++ 성능

```cpp
// ❌ 루프 내 불필요한 복사
for (const auto item : largeVector) { ... }    // 매번 복사

// ✅ 참조 사용
for (const auto& item : largeVector) { ... }   // 복사 없음
```

```cpp
// ❌ 루프 내 동적 할당
for (int i = 0; i < 10000; i++) {
    auto buf = std::make_unique<char[]>(4096);
    process(buf.get());
}

// ✅ 루프 밖 할당, 재사용
auto buf = std::make_unique<char[]>(4096);
for (int i = 0; i < 10000; i++) {
    memset(buf.get(), 0, 4096);
    process(buf.get());
}
```

---

## Part 9: 에러 처리 / 로깅

> **통계**: AI 코드는 에러 무시, 불충분한 예외 처리가 빈번하다.

### 9.1 예외 처리 원칙

```java
// ❌ 예외 삼키기
try {
    riskyOperation();
} catch (Exception e) {
    // 아무것도 안 함
}

// ❌ 포괄적 catch
try {
    riskyOperation();
} catch (Exception e) {
    e.printStackTrace();        // 프로덕션에서 stdout 출력
}

// ✅ 구체적 예외 + 로깅 + 적절한 대응
try {
    riskyOperation();
} catch (IOException e) {
    log.error("파일 처리 실패: {}", filename, e);
    throw new ServiceException("처리 실패", e);     // 또는 적절한 복구
} catch (IllegalArgumentException e) {
    log.warn("잘못된 입력: {}", input, e);
    return defaultValue;
}
```

### 9.2 C/C++ 에러 처리

```cpp
// ❌ 반환값 무시
open("/dev/video0", O_RDONLY);
read(fd, buffer, size);

// ✅ 모든 반환값 검사
int fd = open("/dev/video0", O_RDONLY);
if (fd < 0) {
    LOG_ERROR("Failed to open device: " << strerror(errno));
    return false;
}
ssize_t n = read(fd, buffer, size);
if (n < 0) {
    LOG_ERROR("Read failed: " << strerror(errno));
    close(fd);
    return false;
}
```

### 9.3 타임아웃 필수 설정

```java
// ❌ 타임아웃 없음 (무한 대기 가능)
HttpURLConnection conn = (HttpURLConnection) url.openConnection();

// ✅ 타임아웃 설정
HttpURLConnection conn = (HttpURLConnection) url.openConnection();
conn.setConnectTimeout(5000);   // 5초
conn.setReadTimeout(10000);     // 10초
```

```javascript
// ❌ fetch에 타임아웃 없음
const res = await fetch('/api/data');

// ✅ AbortController로 타임아웃
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 10000);
try {
    const res = await fetch('/api/data', { signal: controller.signal });
} finally {
    clearTimeout(timeoutId);
}
```

---

## Part 10: 코드 품질 / 가독성

> **통계**: AI 코드의 가독성 문제 3배 이상 증가. 네이밍, 포매팅 불일치가 주요 원인.

### 10.1 네이밍 규칙

| 언어 | 변수/함수 | 클래스/타입 | 상수 | 파일 |
|------|---------|-----------|------|------|
| Java | camelCase | PascalCase | UPPER_SNAKE | PascalCase.java |
| C++ | camelCase 또는 snake_case (프로젝트 기존 방식) | PascalCase | UPPER_SNAKE | snake_case.cpp/.h |
| JavaScript | camelCase | PascalCase | UPPER_SNAKE | camelCase.js 또는 kebab-case.js |
| CSS | kebab-case | - | - | kebab-case.css |
| SQL | snake_case | - | - | - |

**단, 프로젝트 기존 컨벤션이 위와 다르면 기존 컨벤션을 따른다.**

### 10.2 Magic Number 금지

```java
// ❌ 의미 불명
if (status == 3) { ... }
Thread.sleep(86400000);

// ✅ 상수 정의
private static final int STATUS_COMPLETED = 3;
private static final long ONE_DAY_MS = 24 * 60 * 60 * 1000L;
if (status == STATUS_COMPLETED) { ... }
Thread.sleep(ONE_DAY_MS);
```

### 10.3 주석 원칙

```java
// ❌ 무의미한 주석 (코드를 반복)
int count = 0;   // count를 0으로 초기화

// ✅ WHY를 설명하는 주석
// MTU(1500) - IP(20) - UDP(8) - RTP Header(80) = 1392
private static final int MAX_RTP_PAYLOAD = 1392;
```

### 10.4 함수 크기

- 한 함수는 **하나의 책임**만 갖는다
- 함수 길이가 50줄을 초과하면 분리를 검토한다
- 중첩 깊이(indent depth)가 3단계를 초과하면 조기 반환(early return)으로 리팩토링

```java
// ❌ 깊은 중첩
void process(Request req) {
    if (req != null) {
        if (req.isValid()) {
            if (req.hasPermission()) {
                doWork(req);
            }
        }
    }
}

// ✅ 조기 반환 (Guard Clause)
void process(Request req) {
    if (req == null) return;
    if (!req.isValid()) return;
    if (!req.hasPermission()) return;
    doWork(req);
}
```

---

## Part 11: CSS 특화 규칙

### 11.1 기본 원칙

```css
/* ❌ !important 남용 */
.button { color: red !important; }

/* ✅ 셀렉터 구체성(specificity)으로 해결 */
.form .button { color: red; }
```

```css
/* ❌ 인라인 스타일 */
<div style="color: red; margin: 10px;">

/* ✅ 클래스 사용 */
<div class="alert-text">
.alert-text { color: red; margin: 10px; }
```

### 11.2 반응형 디자인

```css
/* ❌ 고정 px만 사용 */
.container { width: 1200px; }

/* ✅ 상대 단위 + max-width */
.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* ✅ Mobile First 미디어 쿼리 */
.grid { display: flex; flex-direction: column; }
@media (min-width: 768px) {
    .grid { flex-direction: row; }
}
```

### 11.3 레이아웃

```css
/* ❌ float 기반 레이아웃 (레거시) */
.col { float: left; width: 33.33%; }
.clearfix::after { content: ''; clear: both; display: table; }

/* ✅ Flexbox / Grid */
.container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
```

---

## Part 12: JSP 특화 규칙

### 12.1 스크립틀릿 사용 최소화

```jsp
<%-- ❌ 스크립틀릿 남용 --%>
<%
    String name = request.getParameter("name");
    out.println("<p>" + name + "</p>");
%>

<%-- ✅ EL + JSTL 사용 --%>
<c:set var="name" value="${param.name}" />
<p><c:out value="${name}" /></p>
```

### 12.2 비즈니스 로직 분리

```jsp
<%-- ❌ JSP에서 DB 직접 접근 --%>
<%
    Connection conn = DriverManager.getConnection(...);
    ResultSet rs = conn.createStatement().executeQuery("SELECT * FROM users");
%>

<%-- ✅ MVC 패턴: Controller에서 데이터 준비, JSP는 뷰만 담당 --%>
<%-- Controller: request.setAttribute("users", userService.getAll()); --%>
<c:forEach var="user" items="${users}">
    <tr>
        <td><c:out value="${user.name}" /></td>
    </tr>
</c:forEach>
```

### 12.3 인코딩 설정 필수

```jsp
<%-- 모든 JSP 파일 최상단 --%>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
```

---

## Part 13: AI 통계적 실수 종합 대응표

> 아래 표는 CodeRabbit 보고서 및 다수 연구에서 추출한 AI의 주요 실수 패턴이다.
> 각 항목에 대해 이 가이드라인의 어느 Part에서 대응하는지 명시한다.

| # | AI 실수 유형 | 발생 배율 | 증상 | 대응 Part |
|---|------------|---------|------|----------|
| 1 | **로직/정확성 오류** | 1.75x | 잘못된 조건, 비즈니스 로직 누락 | Part 0, 1 |
| 2 | **존재하지 않는 API 호출** | 매우 빈번 | 컴파일 에러, 런타임 에러 | Part 1 |
| 3 | **Null 체크 누락** | 빈번 | NPE, Segfault, TypeError | Part 5 |
| 4 | **과도한 I/O** | 8x | 느린 응답, DB 과부하 | Part 3, 8 |
| 5 | **보안 취약점** | 1.57x | SQL Injection, XSS, 평문 비밀번호 | Part 7 |
| 6 | **비밀번호 처리 오류** | 1.88x | 평문 저장, 약한 해시 | Part 7.1 |
| 7 | **안전하지 않은 객체 참조** | 1.91x | 권한 없는 데이터 접근 | Part 7.2 |
| 8 | **코드 복제/중복** | 4x | 유지보수 지옥, 불일치 | Part 2 |
| 9 | **가독성/네이밍 불일치** | 3x+ | 리뷰 부담, 컨벤션 불일치 | Part 10 |
| 10 | **동시성 오류** | 빈번 | Race condition, 데드락 | Part 6 |
| 11 | **에러 무시/삼키기** | 빈번 | 조용한 실패, 디버깅 불가 | Part 9 |
| 12 | **성능 문제** | 1.42x | O(n²), 루프 내 할당, DOM 반복 | Part 8 |
| 13 | **리소스 미해제** | 빈번 | 메모리 누수, 커넥션 풀 고갈 | Part 4 |
| 14 | **타임아웃 미설정** | 빈번 | 무한 대기, 서비스 행업 | Part 9.3 |
| 15 | **트랜잭션 범위 과다** | 빈번 | DB 락 경합, 성능 저하 | Part 3.4 |
| 16 | **플랫폼 가정** | C/C++ 빈번 | 타 OS 빌드 실패 | Part 1.3 |
| 17 | **캐시 미활용** | 빈번 | 불필요한 중복 조회 | Part 2.2 |
| 18 | **N+1 쿼리** | 빈번 | DB 부하 폭증 | Part 3.2 |
| 19 | **인덱스 무효화 쿼리** | 빈번 | Full Table Scan | Part 3.3 |
| 20 | **XSS/CSRF 미방어** | 빈번 | 클라이언트 공격 | Part 7.3, 7.4 |

---

## Part 14: 완료 전 최종 체크리스트

코드 작성 후 "완료"를 선언하기 **전에** 아래 항목을 **전부** 확인한다.

### 정확성 검증
- [ ] 새로 사용한 모든 메서드/클래스/함수가 **실제로 존재**하는가?
- [ ] 메서드 시그니처(파라미터 순서, 타입, 반환값)가 **정확**한가?
- [ ] 기존 호출부에 영향을 주는 변경이 있으면 **전부 반영**했는가?
- [ ] 컴파일/빌드 에러 가능성이 **없는가**?

### 성능 검증
- [ ] 루프 안에서 DB/네트워크/파일 I/O를 호출하지 않는가?
- [ ] O(n²) 이상의 복잡도를 **회피**했는가?
- [ ] 불필요한 객체 생성, 문자열 결합이 없는가?
- [ ] 캐시가 있으면 캐시를 **사용**했는가?
- [ ] 대량 데이터는 페이징/스트리밍으로 처리하는가?

### 안전성 검증
- [ ] 모든 사용자 입력에 **검증/이스케이프**를 적용했는가?
- [ ] SQL은 **파라미터 바인딩**으로 작성했는가?
- [ ] null/undefined/nullptr **체크**가 충분한가?
- [ ] 예외를 **삼키지** 않고 적절히 처리했는가?
- [ ] 리소스(Connection, Stream, File)를 **확실히 해제**하는가?
- [ ] 타임아웃이 **설정**되었는가?

### 동시성 검증
- [ ] 공유 자원 접근에 **동기화**가 적용되었는가?
- [ ] 데드락 위험이 **없는가**?
- [ ] 불변 객체를 최대한 활용했는가?

### 품질 검증
- [ ] 기존 네이밍 컨벤션과 **일치**하는가?
- [ ] 중복 코드가 **없는가**?
- [ ] 기존 유틸/헬퍼를 **활용**했는가?
- [ ] 추정 없이 **실제 코드를 확인**하고 작성했는가?

---

## 위반 시 대응

위 가이드라인을 위반한 코드 발견 시:
1. **즉시 수정** — 위반 코드를 삭제하고 가이드라인에 맞게 재작성
2. **위반 원인 분석** — 어떤 Part의 어떤 규칙을 위반했는지 명시
3. **재발 방지** — 해당 패턴이 다른 곳에도 있는지 검색하여 일괄 수정

---

## 커뮤니케이션 규칙

1. 모든 응답, 질문, 설명은 **한글**로 작성
2. 코드 수정 전 **영향 범위**를 먼저 보고
3. 대규모 변경 시 **단계별 계획**을 먼저 제시
4. 불확실한 사항은 **추측하지 말고 질문**

---

**이 가이드라인의 목표**: AI 코딩의 통계적 실패 패턴 90% 이상을 사전 차단하여,
**모든 언어(Java, C/C++, JSP, JavaScript, CSS, SQL)에서 프로덕션 수준의 안전하고 최적화된 코드**를 생성한다.
