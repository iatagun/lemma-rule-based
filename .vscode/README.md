# VS Code Agent Kurulumu

Bu proje, VS Code'da Türkçe Morfolojik Çözümleyici agent'ı olarak kullanılabilir.

## Gereksinimler

1. **VS Code** (en son sürüm)
2. **VS Code Copilot Agent** veya **MCP destekli agent**
3. **Python 3.10+**

## Kurulum

### 1. MCP Server Test

Önce MCP server'ın çalıştığını doğrula:

```powershell
cd lemma-rule-based
python -X utf8 mcp_server/test_server.py
```

### 2. VS Code MCP Yapılandırması

VS Code'da `Command Palette` → `MCP: Configure` seç veya `.vscode/mcp.json` dosyasını oluştur:

```json
{
  "servers": {
    "turkish-morphology": {
      "type": "stdio",
      "command": "python",
      "args": [
        "-X",
        "utf8",
        "-m",
        "mcp_server.server"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8"
      },
      "cwd": "${workspaceFolder}"
    }
  }
}
```

### 3. Agent Talimatları

Proje zaten şu dosyalarla agent'a hazır:

- `.skills/00_overview.md` — Proje özeti
- `.skills/01_phonology.md` — Dilbilgisi kuralları
- `.skills/02_benchmarking.md` — Test prosedürleri
- `.skills/03_adding_suffixes.md` — Yeni ek ekleme
- `.skills/04_debugging.md` — Hata ayıklama
- `.skills/05_coding_standards.md` — Kod standartları
- `.skills/06_suffix_reference.md` — Türetim ekleri referansı
- `AGENTS.md` — Kapsamlı agent talimatları

Agent'ı başlatırken bu dosyaları talimat olarak ver.

## VS Code Tasks

Benchmark ve test çalıştırmak için `Ctrl+Shift+P` → `Tasks: Run Task`:

- **Run Benchmark** — BOUN Treebank benchmark çalıştır
- **Test MCP Server** — MCP server test et

## MCP Tools Kullanımı

Agent, şu araçları kullanabilir:

| Tool | Açıklama |
|------|----------|
| `analyze_word` | Tek sözcük analizi |
| `analyze_all` | Tüm olası çözümlemeler |
| `run_benchmark` | BOUN benchmark çalıştır |
| `check_dictionary` | Sözlük kontrolü |
| `test_suffixes` | Ek eşleşme testi |
| `get_project_info` | Proje bilgileri |

## Örnek Agent Talimatı

Agent'ı başlatırken:

```
Bu proje Türkçe Morfolojik Çözümleyici'dir. Kural-tabanlı olarak
sözcükleri kök ve eklerine ayırır.

Skill dosyalarını oku:
- .skills/00_overview.md (genel bakış)
- .skills/01_phonology.md (dilbilgisi kuralları)
- .skills/06_suffix_reference.md (ek referansı)

MCP server'ı kullanarak analiz yapabilirsin.
```

## Sorun Giderme

### MCP Server Çalışmıyor

```powershell
# Manuel test
python -X utf8 -c "from mcp_server.server import *; print('OK')"
```

### Türkçe Karakter Sorunu

`PYTHONIOENCODING=utf-8` ortam değişkeninin ayarlandığından emin ol.

## Copilot Chat Agent

VS Code Copilot Chat kullanıyorsan, agent talimatları için `AGENTS.md` dosyasını kullan.
