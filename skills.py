#!/usr/bin/env python3
"""
Skill Manager — Agent skill yönetim sistemi

Bu modül, lemma-rule-based projesi için skill dosyalarını yönetir ve
agent'a otomatik olarak gerekli bilgileri sağlar.

Kullanım:
    # Tüm skillleri listele
    from skills import list_skills
    
    # Belirli skill'i yükle  
    content = load_skill("phonology")
    
    # Göreve göre otomatik skill seç
    prompt = create_agent_prompt("elma için lemma çözümle")
    
    # CLI kullanımı
    python skills.py list
    python skills.py load phonology
    python skills.py agent "benchmark test"
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
SKILLS_DIR = PROJECT_ROOT / ".skills"

# Skill dosya mapping'i
SKILL_FILES = {
    # Morfoloji skill'leri
    "overview": "00_overview.md",
    "phonology": "01_phonology.md",
    "benchmarking": "02_benchmarking.md",
    "adding_suffixes": "03_adding_suffixes.md",
    "debugging": "04_debugging.md",
    "coding_standards": "05_coding_standards.md",
    "suffix_reference": "06_suffix_reference.md",
    
    # Dependency parsing
    "dep_parsing": "07_bert_dep_parsing.md",
    
    # Aliases
    "morphology": "00_overview.md",
    "benchmark": "02_benchmarking.md",
    "suffix": "06_suffix_reference.md",
}

# Category mapping
SKILL_CATEGORIES = {
    "morphology": ["00_overview.md", "01_phonology.md", "02_benchmarking.md", "03_adding_suffixes.md", "04_debugging.md", "05_coding_standards.md", "06_suffix_reference.md"],
    "dep_parsing": ["07_bert_dep_parsing.md"],
}


def list_skills() -> list[str]:
    """Tüm mevcut skill isimlerini döndür."""
    return list(SKILL_FILES.keys())


def get_skill_path(name: str) -> Optional[Path]:
    """Skill dosyasının tam yolunu döndür."""
    if name not in SKILL_FILES:
        return None
    path = SKILLS_DIR / SKILL_FILES[name]
    return path if path.exists() else None


def load_skill(name: str, category: bool = False) -> str:
    """
    Skill dosyasını içeriğini yükle.
    
    Args:
        name: Skill adı (örn. "phonology", "dep_parsing")
        category: True ise tüm kategoriyi yükle
    
    Returns:
        Skill içeriği veya hata mesajı
    """
    if name not in SKILL_FILES:
        return f"Hata: '{name}' bilinmiyor. Mevcut: {list_skills()}"
    
    path = get_skill_path(name)
    if not path:
        return f"Hata: Dosya bulunamadı: {path}"
    
    if category and name in SKILL_CATEGORIES:
        # Tüm kategoriyi yükle
        contents = []
        for fname in SKILL_CATEGORIES[name]:
            fpath = SKILLS_DIR / fname
            if fpath.exists():
                contents.append(f"=== {fname} ===\n\n{fpath.read_text(encoding='utf-8')}")
        return "\n\n".join(contents)
    
    return path.read_text(encoding='utf-8')


def load_skill_summary(name: str) -> dict:
    """Skill özetini döndür."""
    content = load_skill(name)
    if content.startswith("Hata:"):
        return {"error": content}
    
    # Başlık ve özet çıkar
    lines = content.split("\n")
    title = lines[0].strip("#").strip() if lines else "Unknown"
    
    # Durum satırını bul
    status = {}
    for line in lines[1:10]:
        if "Durum" in line or "Eğitim" in line or "✓" in line or "○" in line or "⚠" in line:
            status["status_line"] = line.strip()
            break
    
    return {
        "name": name,
        "title": title,
        "file": str(SKILL_FILES.get(name)),
        "status": status.get("status_line", "N/A"),
        "content": content[:500] + "..." if len(content) > 500 else content
    }


def update_skill(name: str, content: str) -> bool:
    """
    Skill dosyasını güncelle.
    
    Args:
        name: Skill adı
        content: Yeni içerik
    
    Returns:
        Başarılı mı?
    """
    path = get_skill_path(name)
    if not path:
        print(f"Hata: Skill bulunamadı: {name}")
        return False
    
    path.write_text(content, encoding='utf-8')
    print(f"Güncellendi: {path}")
    return True


def create_agent_prompt(task: str, content: str = None) -> str:
    """
    Görev için agent prompt'u oluştur.
    
    Args:
        task: Görev açıklaması
        content: Önceden yüklenmiş içerik (opsiyonel)
    
    Returns:
        Agent prompt string'i
    """
    if content is None:
        content = _get_auto_content(task)
    
    return f"Görev: {task}\n\n{content}"


def _get_auto_content(task: str) -> str:
    """Göreve göre otomatik içerik seç."""
    task_lower = task.lower()
    
    if any(k in task_lower for k in ["lemma", "ek", "kök", "morphology", "morfoloj"]):
        return load_skill("morphology", category=True)
    elif any(k in task_lower for k in ["dep", "parsing", "sözdizim", "bağım"]):
        return load_skill("dep_parsing")
    elif any(k in task_lower for k in ["benchmark", "test", "doğruluk"]):
        return load_skill("benchmarking")
    elif any(k in task_lower for k in ["debug", "hata", "sorun"]):
        return load_skill("debugging")
    else:
        return load_skill("overview")


# ── CLI Interface ─────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Skill Loader")
    parser.add_argument("command", nargs="?", help="Komut: list, load, update, agent")
    parser.add_argument("skill", nargs="?", help="Skill adı")
    parser.add_argument("--category", "-c", action="store_true", help="Tüm kategoriyi yükle")
    parser.add_argument("--preview", "-p", action="store_true", help="Önizleme göster")
    args = parser.parse_args()
    
    if args.command == "list":
        print("Mevcut skilller:")
        for s in list_skills():
            print(f"  - {s}")
    
    elif args.command == "load" and args.skill:
        if args.preview:
            summary = load_skill_summary(args.skill)
            print(f"Title: {summary.get('title', 'N/A')}")
            print(f"Status: {summary.get('status', 'N/A')}")
            print(f"File: {summary.get('file', 'N/A')}")
        else:
            print(load_skill(args.skill, category=args.category))
    
    elif args.command == "agent":
        task = args.skill or input("Görev > ")
        print(create_agent_prompt(task))
    
    else:
        print("Komutlar: list, load <skill>, agent <task>")
        print("Örnek:")
        print("  python skills.py list")
        print("  python skills.py load phonology")
        print("  python skills.py load dep_parsing --category")
        print("  python skills.py agent 'elma için morfoloji analizi yap'")


if __name__ == "__main__":
    main()