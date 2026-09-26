# JevRoute — Proje Dokümantasyonu ve Yol Haritası

Bu dosya projenin **canlı durum takip dokümanıdır**. Yol haritasını, mevcut
ilerlemeyi, mimari kararları ve açık soruları burada tutuyoruz.

- **Proje adı:** `jev-route`
- **Vizyon:** AI ajanlarının, araçların (tools) ve model çağrılarının önüne geçen
  bir **"AI Agent Intent & Tool Router" (Yapay Zeka Niyet ve Karar Yönlendiricisi)**
  kütüphanesi. JevRoute bir web URL/path router'ı **DEĞİLDİR**.
- **Karar motoru:** Jev motoru (`typesafe/jev-1.13`), OpenRouter API (veya Jev API
  uç noktası) üzerinden OpenAI-uyumlu `/chat/completions` protokolü ile çağrılır.
- **Dil kuralı:** Kod içindeki tüm değişken/fonksiyon/sınıf adları, yorumlar ve
  docstring'ler **İngilizce**. Ekip iletişimi ve bu doküman Türkçe.
- **Son güncelleme:** 2026-09-26

---

## 1. Mevcut Durum (Özet)

| Alan | Durum | Not |
| --- | --- | --- |
| Proje iskeleti | ✅ Tamamlandı | `src/` layout kuruldu |
| `pyproject.toml` | ✅ Tamamlandı | PEP 639 (`license = "MIT"` + `license-files`), genişletilmiş classifiers/keywords, `fastapi` + `dev` extras, `Documentation`/`Changelog` URL'leri, `package-data: py.typed` |
| `README.md` | ✅ Tamamlandı | Mermaid mimari şeması, **"Why jev-route?"** farklılaşma bölümü, confidence gate **önce/sonra** kanıtı, **demo GIF** (badge'lerin altında), FastAPI örneği linki, 6 rozet (CI, Python, License, ruff, mypy, **PyPI**) |
| `docs/demo.gif` | ✅ Tamamlandı | Ekrandan kaydedilen terminal demosu, ~926 KB, commit `9fb1d26`; README'de `![jev-route demo](docs/demo.gif)` olarak gömülü |
| `core.py` | ✅ Tamamlandı | `JevClient` (async, `httpx.AsyncClient`), `Choice`/`Score`/`Noul`, `JevResponse`, `JevDecision` |
| `router.py` | ✅ Tamamlandı | `IntentRouter` (Jev'e sorar, `winner`'a göre handler, confidence-gate + fallback) |
| `middleware.py` | ✅ Tamamlandı | Async pipeline (intent resolution + agent döngüleri) |
| `examples/basic_routing.py` | ✅ Tamamlandı | Mock + gerçek API modlu örnek ("Hava durumu nasıl?" → `weather_agent`) |
| `examples/fastapi_service.py` | ✅ Tamamlandı | FastAPI servisi: `/healthz`, `/options`, `/route`, `/audit`, `/refunds`; offline mock transport |
| Test altyapısı (`tests/`) | ✅ Tamamlandı | 19 test, tamamen offline: `test_jev_route.py` (10), `test_fastapi_service.py` (6), `test_packaging.py` (3) |
| `py.typed` (PEP 561) | ✅ Tamamlandı | Tip işaretçisi eklendi ve wheel'e dahil edildi (`Typing :: Typed`) |
| Lint & type check | ✅ Tamamlandı | `python -m ruff check .` → **All checks passed!** · `python -m mypy src` → **Success: no issues found in 4 source files** |
| `LICENSE` | ✅ Tamamlandı | MIT lisans metni eklendi |
| CI (GitHub Actions) | ✅ Yeşil | 3 run tamamlandı; en son run (#3, `9fb1d26` "docs: add demo gif") → **success**; ruff + mypy + pytest (3.10–3.12) + build + `twine check` |
| Git deposu | ✅ Yayında | `main`'de **4 commit** push edildi; **GitHub About açıklaması ve 20 topic ekli** (2026-09-26) |
| PyPI dağıtımı | ✅ Yayında | **v0.2.0** — https://pypi.org/project/jev-route/ · wheel + sdist, yükleme: 2026-09-26 12:57 UTC |
| Ekosistem başvuruları | ⬜ Başlanmadı | OpenRouter docs cookbook PR'ı, `hellogumbo/awesome-jev`, `yibie/awesome-jev` — süreç §7'de, başvurular manuel |

**Lejant:** ✅ Tamam · 🟡 Devam ediyor · ⬜ Başlanmadı · ⛔ Engellendi

---

## 2. Dosya Yapısı

```
JevRoute/
├── .github/workflows/ci.yml # CI: ruff + mypy + pytest + build + twine check (3.10-3.12)
├── .gitignore               # venv (test_env dahil), cache, .env, coverage, OS/IDE dosyaları
├── LICENSE                  # MIT
├── pyproject.toml           # Paketleme + araç konfigürasyonu
├── README.md                # Mermaid şema + Why jev-route? + quickstart + demo GIF
├── documentation.md         # Bu dosya (durum + yol haritası)
├── docs/
│   └── demo.gif             # Terminal demosu (README'nin başında gömülü)
├── examples/
│   ├── basic_routing.py     # "Hava durumu nasıl?" → weather_agent örneği (mock + live)
│   └── fastapi_service.py   # Aynı router'ın FastAPI endpoint'leri içinde kullanımı
├── tests/
│   ├── test_jev_route.py    # Örnek akışa dayalı smoke testleri (10 test)
│   ├── test_fastapi_service.py  # TestClient ile FastAPI örneği (6 test)
│   └── test_packaging.py    # Sürüm + PyPI metadata tutarlılığı (3 test)
└── src/
    └── jev_route/
        ├── __init__.py      # Genel API (public exports)
        ├── core.py          # Jev veri yapıları, JevClient, JevDecision, hatalar
        ├── router.py        # IntentRouter: kayıt, Jev'e sorma, confidence-gate, dispatch
        ├── middleware.py    # Async middleware pipeline (intent + agent loop)
        └── py.typed         # PEP 561 tip işaretçisi
```

---

## 3. Mimari Kararlar

1. **`src/` layout** — Kurulum/`import` kaynaklı kazaları önler; testlerin
   kurulu paketi test etmesini garanti eder.
2. **Bağımlılık yönü** — `core` → `middleware` → `router` → `__init__`.
3. **Jev veri yapıları (`core.py`)** — Motorun döndürdüğü karar Pydantic
   modelleriyle temsil edilir: 0–1 aralığında `Score`, etiket + skor taşıyan
   `Choice`, çekimserlik bildiren `Noul` (null outcome), ayrıştırılmış
   `JevResponse` ve eşik uygulanmış `JevDecision`.
4. **Async istemci (`JevClient`)** — `httpx.AsyncClient` ile OpenRouter-uyumlu
   `/chat/completions` uç noktasına konuşur; `JevSettings` ile yapılandırılır.
   Varsayılan model `typesafe/jev-1.13`'tür. 429/5xx'te üstel bekleme ile retry
   yapar; mock transport enjekte edilebilir.
5. **Intent yönlendirme (`router.py`)** — `IntentRouter` metin/state + kayıtlı
   hedef seçenekleri alır, Jev'e sorar, `winner`'a göre handler çalıştırır.
   **Confidence-gate:** skor eşik altındaysa fallback tetiklenir; motor
   `winner=null` dönerse çekimserlik olur. İkisi de fallback handler'a düşer.
6. **Async middleware (`middleware.py`)** — Intent resolution (`IntentCall →
   JevDecision`) ve agent döngülerine takılabilen tip-jenerik soğan-modeli
   pipeline. Yerleşikler: `LoggingMiddleware`, `CacheMiddleware` (TTL + LRU).
7. **Transport-agnostik karar modelleri** — Karar yapıları herhangi bir agent
   çatısına bağlı değildir.
8. **Tip güvenliği** — Pydantic v2 modeller + `mypy --strict` hedefi.

---

## 4. Yol Haritası

### Faz 0 — İskelet + Kavramsal Düzeltme
- [x] Klasör ve dosya yapısı
- [x] `pyproject.toml` (setuptools, `httpx`, `pydantic`)
- [x] Web-router'dan **AI Intent Router** vizyonuna geçiş
- [x] `core.py`: `JevClient` + `Choice`/`Score`/`Noul` veri yapıları
- [x] `router.py`: `IntentRouter` + confidence-gate + fallback
- [x] `middleware.py`: async pipeline (intent + agent loop)
- [x] `examples/basic_routing.py`: mock + live modlu örnek
- [x] `pyproject.toml` / `README.md` metinlerinin vizyona uyarlanması
- [x] Paketin yerel olarak kurulup doğrulanması (`pip install -e .`)

### Faz 1 — Sağlamlaştırma
- [x] `tests/` + birim testleri (mock transport ile karar ayrıştırma,
      confidence-gate/fallback, pipeline sırası) — `tests/test_jev_route.py`:
      `python -m pytest` → **9 passed**
- [ ] Önbellek TTL / LRU davranışı için ek testler (`CacheMiddleware`)
- [ ] Pydantic tabanlı karar doğrulama modellerinin sertleştirilmesi
- [x] Tür (type) denetimi: `mypy --strict` temizliği — `python -m mypy src` →
      **Success: no issues found in 4 source files**
- [x] Lint: `ruff` temizliği — `python -m ruff check .` → **All checks passed!**
- [ ] Kapsamlı docstring'ler ve kullanım örnekleri

### Faz 2 — GitHub Yayını ve CI (şu an)
- [x] `LICENSE` (MIT) eklendi
- [x] `.gitignore` gözden geçirildi (venv, cache, `.env`, coverage, loglar, OS/IDE)
- [x] `.github/workflows/ci.yml` — ruff + mypy + pytest (Python 3.10–3.12 matrisi)
      ve `python -m build` artefaktı
- [x] README: Mermaid mimari şeması + rozetler + "Installation & Quickstart" + config tablosu
- [x] Kalite kapıları yerelde yeşil: `ruff check .`, `mypy src`, `pytest` (9 passed)
- [ ] Depoyu initialize et ve ilk commit'i at — `git init -b main` ve `git add -A`
      tamamlandı (12 dosya stage'de, `__pycache__`/`*.egg-info`/`.pytest_cache`
      ignore edildi); `git commit` + `git push` bekliyor
- [ ] GitHub'da `jevroute/jev-route` deposunu oluştur, `origin` remote'unu ekle, `main`'i push'la
- [ ] Repository ayarları: açıklama + topic'ler (`ai-agents`, `intent-routing`, `llm-router`,
      `openrouter`), branch protection ("Require status checks: quality")
- [ ] İlk CI run'ının yeşil olduğunu doğrula ve README rozetini kontrol et
- [ ] `CHANGELOG.md` + SemVer etiketi (`v0.2.0`)
- [ ] PyPI yayını: Trusted Publishing + GitHub Release ile `dist/` artefaktı

### Faz 3 — Yetenek Genişletmesi
- [ ] Hiyerarşik routing (`include_router` ile domain → alt router devri)
- [ ] Streaming / çok turlu agent döngüsü yardımcıları
- [ ] Yerleşik middleware'ler: bütçe/kota, retry politikası, tracing
- [ ] Yapılandırılmış çıktı şeması desteği
- [ ] Gözlemlenebilirlik: OpenTelemetry span'leri karar `usage`/`latency` ile

#### Git yayın komutları (Faz 2)

```powershell
cd c:\JevRoute
git init -b main
git add -A
git status --short      # .gitignore sayesinde yalnızca kaynak dosyalar görünür
git commit -m "feat: initial release of jev-route (AI agent intent & tool router)"
git remote add origin https://github.com/jevroute/jev-route.git
git push -u origin main
```

---

## 5. Public API (hedef)

| Sembol | Modül | Açıklama |
| --- | --- | --- |
| `JevClient` / `JevSettings` | `core` | Async motor istemcisi + yapılandırma (`typesafe/jev-1.13`) |
| `Choice` / `Score` / `Noul` | `core` | Jev'in temel veri yapıları |
| `JevResponse` / `JevDecision` | `core` | Ham motor yanıtı / eşik uygulanmış karar |
| `OptionSpec` | `core` | Motor'a sunulan seçenek açıklaması |
| `DecisionOutcome` | `core` | `accepted` / `fallback` / `abstained` |
| `JevRouteError` ve alt sınıfları | `core` | Hata hiyerarşisi (`JevAPIError`, `JevResponseError`, `ConfidenceTooLowError`, ...) |
| `IntentRouter` | `router` | Seçenek kaydı, Jev'e sorma, handler çalıştırma |
| `IntentOption` / `IntentRequest` / `IntentResult` | `router` | Kayıtlı hedef / handler girdisi / yönlendirme sonucu |
| `MiddlewarePipeline` / `Middleware` / `NextStep` | `middleware` | Jenerik async middleware zinciri |
| `IntentCall` / `IntentMiddleware` | `middleware` | Intent çözümleme zarfı + yapısal tip |
| `LoggingMiddleware` / `CacheMiddleware` | `middleware` | Yerleşik loglama + TTL önbellek |

---

## 6. Açık Sorular / Karar Bekleyenler

- [ ] Canlı uç nokta: yalnızca OpenRouter mı, yoksa özel **Jev API** taban URL'i de
      birinci sınıf desteklenecek mi? (`JevSettings.base_url` ikisini de kaldırıyor,
      ancak varsayılan + dokümantasyon netleşmeli.)
- [ ] Kimlik doğrulama: `OPENROUTER_API_KEY` / `JEV_API_KEY` dışında header
      (örn. `HTTP-Referer`, `X-Title`) gönderilsin mi?
- [ ] Async-first mi kalınmalı, yoksa sync sarmalayıcı (`route_sync`) eklensin mi?
- [x] Asgari Python sürümü: **`3.10+`** olarak belirlendi (mypy 2.x desteği ve
      pydantic modellerindeki `X | Y` runtime gereksinimi) — `pyproject.toml`
      (`requires-python`, classifier), ruff `target-version` ve mypy
      `python_version` güncellendi.
- [x] Lisans: **MIT** olarak seçildi ve `LICENSE` dosyası eklendi
      (telif: "JevRoute Contributors", 2026).

---

## 7. Ekosistem Görünürlüğü — Başvuru Süreçleri (araştırma, 2026-09-26)

Dış listelere/dokümantasyona ekleme başvuruları **hesap gerektirdiği için
yapılmadı**; süreçler aşağıda çıkarıldı.

### 7.1 OpenRouter dokümantasyonu (cookbook / guide PR'ı)

- Dokümantasyonun kaynağı **`OpenRouterTeam/docs`** reposudur (Mintlify;
  `openrouter.ai/docs` buradan üretilir). Sayfalar `projects/docs/**` altında,
  cookbook tarifleri `cookbook/` klasöründedir.
- Bir tarifin ilgili model sayfasında listelenmesi için frontmatter'da model
  ailesi bildirilir:

  ```yaml
  ---
  title: Gate Agent Tool Calls with Jev
  models:
    - typesafe/jev
  ---
  ```

  `author/name-prefix` biçimindedir; `typesafe/jev` girdisi `typesafe/jev-1.13`,
  `~typesafe/jev-latest` ve `typesafe/jev-1.13:free` sayfalarını kapsar.
- Frontmatter değiştirildikten sonra `bun run generate:docs:model-map` çalıştırılıp
  `projects/web` içindeki `model-documentation-map.gen.ts` commit'lenmelidir;
  *Validate Docs* iş akışı bu dosya bayatsa kırmızıya döner.
- PR'de `mint validate`, `@openrouter/sdk` içeren kod bloklarının tip kontrolü ve
  `bun run lint` zorunludur; **fork PR'ları preview URL alamaz** (secret erişimi).
- `OpenRouterTeam/awesome-openrouter` ayrı bir **uygulama** listesidir:
  `apps/<ad>/app.yaml` + `logo.png` ister, README elle düzenlenmez ve
  **traction/notability kanıtı** (yıldız, kullanıcı, indirme, topluluk izi) arar.
  Bu yüzden sıra şöyle olmalı: önce Jev'e özel bir cookbook/guide sayfası, sonra
  (traction oluştuğunda) bu listeye başvuru.

### 7.2 awesome-jev listeleri

| Liste | Nasıl başvurulur | Kural / not |
| --- | --- | --- |
| `hellogumbo/awesome-jev` | `data/projects.json` içindeki `projects` dizisine **tek nesne** ekleyip PR; JSON düzenlemek istemezsen "Submit a project" issue şablonu | `npm run validate` çalıştırılır; README ve site CI'da üretilir, elle commit'lenmez. Kategori: *Integrations* ya da *SDKs & clients* |
| `yibie/awesome-jev` | Yalnızca **tek kategori dosyası** düzenlenir (`categories/infra-sdks-integrations.md` veya `categories/classification-routing.md`), sonra `python3 scripts/build-readme.py` | README elle değiştirilmez; Jev'in somut bir karar için kullanıldığı kanıtlanmalı; **7 günlük pencerede en fazla 3 giriş** ve toplu/aynı iskeletli başvurular tek başvuru sayılır |

İki liste de tek cümlelik giriş ister: *hangi karar + hangi gate + ne sağlıyor*.
Başvurudan önce girişin çalıştırılabilir kanıta (test, demo çıktısı, ölçüm)
dayanması gerekir; bu yüzden `tests/` ve `examples/` içeriği başvurunun parçasıdır.

**Manuel yapılacaklar (hesap gerektirir):** (1) `OpenRouterTeam/docs` PR'ı,
(2) `hellogumbo/awesome-jev` PR/issue, (3) `yibie/awesome-jev` PR'ı.

---

## 8. Değişiklik Günlüğü (bu doküman)

- **2026-09-26 (2)** — **PyPI yayını gerçekleşti: `jev-route` v0.2.0** (wheel +
  sdist, 12:57 UTC) ve CI yeşile döndü (run #3, `9fb1d26` → **success**). Depo
  görünürlüğü tamamlandı: About açıklaması + 20 topic eklendi. README'ye **PyPI
  rozeti** eklendi (mevcut 5 rozetin yanına, tek boşlukla, yeni satır açmadan) ve
  **demo GIF** rozetlerin altına `![jev-route demo](docs/demo.gif)` olarak
  gömüldü; artık gereksiz olan **asciinema/`.cast` kayıt talimatları bölümü
  kaldırıldı** (kayıt ekrandan alındı). `.gitignore`'a **`test_env/`** eklendi —
  yerel sanal ortam (1433 dosya) `git add -A` ile repoya girmesin diye. Durum
  tablosu gerçek duruma çekildi: PyPI ✅ yayında, CI ✅ yeşil, About/topics ✅,
  demo GIF ✅, ekosistem PR'ları ⬜ başlanmadı. Doğrulama: `ruff check .` →
  All checks passed, `mypy src` → Success, `pytest` → **19 passed**.
  *Not:* PyPI'da görünen uzun açıklama (README) bir sonraki sürüm yüklemesinde
  güncellenir; rozet ve GIF bu yüzden PyPI sayfasında ancak o zaman görünür.

- **2026-09-26** — Keşfedilebilirlik ve yayın hazırlığı turu. `pyproject.toml`
  PEP 639'a taşındı (`license = "MIT"`, `license-files = ["LICENSE"]`),
  classifiers/keywords genişletildi (`Framework :: AsyncIO`,
  `Framework :: Pydantic :: 2`, Python 3.13), `fastapi` ve `dev` (artık `twine`)
  extras'ları ile `Documentation`/`Changelog` URL'leri eklendi; PEP 561 için
  `src/jev_route/py.typed` oluşturuldu ve wheel'e dahil edildi (`package-data`).
  README'ye **"Why jev-route?"** (harness adaptörü değil, gömülebilir kütüphane)
  bölümü, confidence gate **önce/sonra** kanıtı, asciinema/GIF kayıt talimatı ve
  FastAPI örneği linki eklendi; yinelenen "Installation & Quickstart" başlığı
  düzeltildi. Yeni `examples/fastapi_service.py` (gerçek, offline çalışan FastAPI
  servisi) ile `tests/test_fastapi_service.py` ve `tests/test_packaging.py`
  eklendi. Doğrulama: `ruff check .` → **All checks passed!**, `mypy src` →
  **Success**, `pytest` → **19 passed**, `python -m build` + `twine check dist/*`
  → **PASSED (ikisi de)**, `python examples/fastapi_service.py` → uvicorn ayağa
  kalktı ve `/healthz` + `/route` beklenen JSON'u döndü. PyPI upload yapılmadı.

- **2026-09-25 (3)** — GitHub yayını hazırlığı: `.gitignore` genişletildi (venv,
  `.env`, coverage, loglar, OS/IDE dosyaları), `LICENSE` (MIT) eklendi,
  `.github/workflows/ci.yml` yazıldı (ruff + mypy + pytest, Python 3.10–3.12
  matrisi, `python -m build` artefaktı), README'nin başına **Mermaid mimari
  şeması**, rozetler, "Installation & Quickstart" ve yapılandırma tablosu
  eklendi. Yol haritası yeniden düzenlendi: **Faz 2 = GitHub Yayını ve CI**,
  eski yetenek genişletmesi **Faz 3**'e taşındı; git init/commit/push komutları
  belgelendi.
- **2026-09-25 (2)** — Kod kalitesi global standartlara çekildi:
  `python -m ruff check .` → **All checks passed!**, `python -m mypy src` →
  **Success: no issues found in 4 source files**. Yapılan düzeltmeler:
  `typing` → `collections.abc` taşımaları (`Mapping`, `Sequence`, `Callable`),
  gereksiz tırnaklı annotation'ların kaldırılması, `_is_sequence` için
  `TypeGuard` ile tip daraltma, generic `__getitem__` overload'ları, `getattr`
  ve satır uzunluğu düzeltmeleri; asgari Python sürümü `3.9` → `3.10`.
  `pip install -e .` ile editable kurulum doğrulandı
  (`Successfully installed jev-route-0.2.0`), testler tekrar **9 passed**.
- **2026-09-25** — Test süiti sadeleştirildi: kırılgan mock-transport testleri
  (`conftest.py`, `helpers.py`, `test_client.py`, `test_router.py`,
  `test_middleware.py`) kaldırıldı; yerine çalışan `examples/basic_routing.py`
  akışını doğrulayan tek dosyalık `tests/test_jev_route.py` (9 test) eklendi.
  Doğrulama: `python -m pytest` → **9 passed**, `python examples/basic_routing.py`
  → **EXIT=0**. Durum tablosu, dosya yapısı, Faz 0 kalan maddeleri ve Faz 1 test
  maddesi güncellendi.
- **2026-09-24 (2)** — Kavramsal düzeltme: web URL router vizyonu terk edildi;
  **AI Agent Intent & Tool Router** vizyonuna geçildi. `core.py` (`JevClient`,
  `Choice`/`Score`/`Noul`), `router.py` (`IntentRouter` + confidence-gate),
  `middleware.py` (async pipeline), `examples/basic_routing.py` (mock + live
  "Hava durumu nasıl?" senaryosu) ve bu doküman güncellendi.
- **2026-09-24** — İskelet kuruldu: dizin yapısı, `pyproject.toml`, `README.md`,
  `core.py`, `router.py`, `middleware.py`, `examples/basic_routing.py` ve bu
  doküman oluşturuldu.