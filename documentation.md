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
- **Son güncelleme:** 2026-09-25

---

## 1. Mevcut Durum (Özet)

| Alan | Durum | Not |
| --- | --- | --- |
| Proje iskeleti | ✅ Tamamlandı | `src/` layout kuruldu |
| `pyproject.toml` | ✅ Tamamlandı | Intent-router açıklaması, `requires-python = ">=3.10"`, ruff/mypy/pytest ayarları |
| `README.md` | ✅ Tamamlandı | Intent-router hızlı başlangıcı |
| `core.py` | ✅ Tamamlandı | `JevClient` (async, `httpx.AsyncClient`), `Choice`/`Score`/`Noul`, `JevResponse`, `JevDecision` |
| `router.py` | ✅ Tamamlandı | `IntentRouter` (Jev'e sorar, `winner`'a göre handler, confidence-gate + fallback) |
| `middleware.py` | ✅ Tamamlandı | Async pipeline (intent resolution + agent döngüleri) |
| `examples/basic_routing.py` | ✅ Tamamlandı | Mock + gerçek API modlu örnek ("Hava durumu nasıl?" → `weather_agent`) |
| Test altyapısı (`tests/`) | ✅ Tamamlandı | `tests/test_jev_route.py` — 9 test, tamamen offline (mock transport); `python -m pytest` → 9 passed |
| Lint & type check | ✅ Tamamlandı | `python -m ruff check .` → **All checks passed!** · `python -m mypy src` → **Success: no issues found in 4 source files** |
| `LICENSE` | ✅ Tamamlandı | MIT lisans metni eklendi |
| CI (GitHub Actions) | 🟡 Hazır (yerel) | `.github/workflows/ci.yml` yazıldı; ilk push sonrası ilk run doğrulanacak |
| Git deposu | 🟡 Başlatıldı | `git init -b main` + `git add -A` yapıldı (12 dosya stage'de); `git commit` bekliyor |
| Yayınlama (PyPI) | ⬜ Başlanmadı | `python -m build` CI'da hazır; etiketleme/release bekliyor |

**Lejant:** ✅ Tamam · 🟡 Devam ediyor · ⬜ Başlanmadı · ⛔ Engellendi

---

## 2. Dosya Yapısı

```
JevRoute/
├── .github/workflows/ci.yml # CI: ruff + mypy + pytest + build (Python 3.10-3.12)
├── .gitignore               # venv, cache, .env, coverage, OS/IDE dosyaları
├── LICENSE                  # MIT
├── pyproject.toml           # Paketleme + araç konfigürasyonu
├── README.md                # Mimari şema (Mermaid) + Installation & Quickstart
├── documentation.md         # Bu dosya (durum + yol haritası)
├── examples/
│   └── basic_routing.py     # "Hava durumu nasıl?" → weather_agent örneği (mock + live)
├── tests/
│   └── test_jev_route.py    # Basit smoke testleri (pytest + mock transport, 9 test)
└── src/
    └── jev_route/
        ├── __init__.py      # Genel API (public exports)
        ├── core.py          # Jev veri yapıları, JevClient, JevDecision, hatalar
        ├── router.py        # IntentRouter: kayıt, Jev'e sorma, confidence-gate, dispatch
        └── middleware.py    # Async middleware pipeline (intent + agent loop)
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

## 7. Değişiklik Günlüğü (bu doküman)

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