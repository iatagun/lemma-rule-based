# MCP Server — Turkish Morphological Analyzer

Model Context Protocol (MCP) server for Turkish morphological analysis.

## Kurulum

### Gereksinimler

```powershell
pip install mcp>=1.0.0
```

### Yapılandırma

Bu dizini projenizin kök dizinine ekleyin veya sembolik bağ oluşturun.

## Kullanım

### VS Code / Cursor MCP Yapılandırması

`.vscode/mcp.json` dosyası oluşturun:

```json
{
  "mcpServers": {
    "turkish-morphology": {
      "command": "python",
      "args": [
        "-X", "utf8", "-m", "mcp_server.server"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

### Manuel Test

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 -m mcp_server.server
```

## Araçlar

### analyze_word
Tek bir sözcüğü analiz eder.

```python
{
  "word": "evlerinden",
  "upos": "NOUN"  // isteğe bağlı
}
```

### analyze_all
Tüm olası analizleri döndürür.

```python
{
  "word": "gelirin",
  "upos": "VERB",
  "max_results": 5
}
```

### run_benchmark
BOUN Treebank benchmark çalıştırır.

```python
{
  "summary_only": true
}
```

### check_dictionary
Sözcüğün sözlükte olup olmadığını kontrol eder.

```python
{
  "word": "kitap",
  "find_root": true
}
```

### test_suffixes
Ek eşleşmelerini test eder.

```python
{
  "word": "geldim",
  "upos": "VERB"
}
```

## Örnek Çıktı

```
Word: evlerinden
Stem: ev
Lemma: ev
Parts: ev + leri + nden
Suffixes:
  leri → İYELİK_3Ç
  nden → AYRILMA
```
