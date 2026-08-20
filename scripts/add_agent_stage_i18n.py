"""Add agent-stage progress keys to all WebUI i18n files (idempotent)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
I18N_DIR = ROOT / "webui" / "i18n"

STAGE_KEYS = [
    "Agent Stage Intelligence",
    "Agent Stage Research",
    "Agent Stage Analysis",
    "Agent Stage Strategy",
    "Agent Stage Hooks",
    "Agent Stage Narrative",
    "Agent Stage Script",
    "Agent Stage Script Revision",
    "Agent Stage Visuals",
    "Agent Stage Titles",
    "Agent Stage QA",
]

TRANSLATIONS = {
    "en": {
        "Agent Stage Intelligence": "Analyzing the niche & audience",
        "Agent Stage Research": "Researching the topic",
        "Agent Stage Analysis": "Analyzing the topic",
        "Agent Stage Strategy": "Building the content strategy",
        "Agent Stage Hooks": "Crafting opening hooks",
        "Agent Stage Narrative": "Planning the narrative",
        "Agent Stage Script": "Writing the script",
        "Agent Stage Script Revision": "Revising the script (critic round)",
        "Agent Stage Visuals": "Planning scenes & visuals",
        "Agent Stage Titles": "Generating titles & thumbnail",
        "Agent Stage QA": "Running quality assurance",
    },
    "zh": {
        "Agent Stage Intelligence": "分析赛道与受众",
        "Agent Stage Research": "研究检索中",
        "Agent Stage Analysis": "选题分析中",
        "Agent Stage Strategy": "制定内容策略",
        "Agent Stage Hooks": "构思开场钩子",
        "Agent Stage Narrative": "规划叙事结构",
        "Agent Stage Script": "撰写文案中",
        "Agent Stage Script Revision": "文案修订（评审轮）",
        "Agent Stage Visuals": "规划画面与场景",
        "Agent Stage Titles": "生成标题与封面",
        "Agent Stage QA": "质量审查中",
    },
    "de": {
        "Agent Stage Intelligence": "Nische & Publikum analysieren",
        "Agent Stage Research": "Thema recherchieren",
        "Agent Stage Analysis": "Thema analysieren",
        "Agent Stage Strategy": "Inhaltsstrategie aufbauen",
        "Agent Stage Hooks": "Aufhänger entwickeln",
        "Agent Stage Narrative": "Erzählstruktur planen",
        "Agent Stage Script": "Skript schreiben",
        "Agent Stage Script Revision": "Skript überarbeiten (Kritikrunde)",
        "Agent Stage Visuals": "Szenen & Bilder planen",
        "Agent Stage Titles": "Titel & Thumbnail erzeugen",
        "Agent Stage QA": "Qualitätsprüfung läuft",
    },
    "es": {
        "Agent Stage Intelligence": "Analizando nicho y audiencia",
        "Agent Stage Research": "Investigando el tema",
        "Agent Stage Analysis": "Analizando el tema",
        "Agent Stage Strategy": "Construyendo la estrategia",
        "Agent Stage Hooks": "Creando ganchos de apertura",
        "Agent Stage Narrative": "Planificando la narrativa",
        "Agent Stage Script": "Escribiendo el guion",
        "Agent Stage Script Revision": "Revisando el guion (ronda crítica)",
        "Agent Stage Visuals": "Planificando escenas y visuales",
        "Agent Stage Titles": "Generando títulos y miniatura",
        "Agent Stage QA": "Ejecutando control de calidad",
    },
    "id": {
        "Agent Stage Intelligence": "Menganalisis niche & audiens",
        "Agent Stage Research": "Meneliti topik",
        "Agent Stage Analysis": "Menganalisis topik",
        "Agent Stage Strategy": "Menyusun strategi konten",
        "Agent Stage Hooks": "Membuat hook pembuka",
        "Agent Stage Narrative": "Merencanakan narasi",
        "Agent Stage Script": "Menulis naskah",
        "Agent Stage Script Revision": "Merevisi naskah (putaran kritik)",
        "Agent Stage Visuals": "Merencanakan scene & visual",
        "Agent Stage Titles": "Membuat judul & thumbnail",
        "Agent Stage QA": "Menjalankan pemeriksaan kualitas",
    },
    "pt": {
        "Agent Stage Intelligence": "Analisando nicho e público",
        "Agent Stage Research": "Pesquisando o tópico",
        "Agent Stage Analysis": "Analisando o tópico",
        "Agent Stage Strategy": "Construindo a estratégia",
        "Agent Stage Hooks": "Criando ganchos de abertura",
        "Agent Stage Narrative": "Planejando a narrativa",
        "Agent Stage Script": "Escrevendo o roteiro",
        "Agent Stage Script Revision": "Revisando o roteiro (rodada crítica)",
        "Agent Stage Visuals": "Planejando cenas e visuais",
        "Agent Stage Titles": "Gerando títulos e miniatura",
        "Agent Stage QA": "Executando controle de qualidade",
    },
    "ru": {
        "Agent Stage Intelligence": "Анализ ниши и аудитории",
        "Agent Stage Research": "Исследование темы",
        "Agent Stage Analysis": "Анализ темы",
        "Agent Stage Strategy": "Построение контент-стратегии",
        "Agent Stage Hooks": "Создание хуков",
        "Agent Stage Narrative": "Планирование нарратива",
        "Agent Stage Script": "Написание сценария",
        "Agent Stage Script Revision": "Правка сценария (раунд критика)",
        "Agent Stage Visuals": "Планирование сцен и визуала",
        "Agent Stage Titles": "Генерация заголовков и обложки",
        "Agent Stage QA": "Проверка качества",
    },
    "tr": {
        "Agent Stage Intelligence": "Niş ve kitle analiz ediliyor",
        "Agent Stage Research": "Konu araştırılıyor",
        "Agent Stage Analysis": "Konu analiz ediliyor",
        "Agent Stage Strategy": "İçerik stratejisi oluşturuluyor",
        "Agent Stage Hooks": "Açılış kancaları oluşturuluyor",
        "Agent Stage Narrative": "Anlatı planlanıyor",
        "Agent Stage Script": "Senaryo yazılıyor",
        "Agent Stage Script Revision": "Senaryo revize ediliyor (kritik turu)",
        "Agent Stage Visuals": "Sahneler ve görseller planlanıyor",
        "Agent Stage Titles": "Başlıklar ve küçük resim oluşturuluyor",
        "Agent Stage QA": "Kalite kontrol çalışıyor",
    },
    "vi": {
        "Agent Stage Intelligence": "Đang phân tích niche & khán giả",
        "Agent Stage Research": "Đang nghiên cứu chủ đề",
        "Agent Stage Analysis": "Đang phân tích chủ đề",
        "Agent Stage Strategy": "Đang xây dựng chiến lược nội dung",
        "Agent Stage Hooks": "Đang tạo câu mở đầu",
        "Agent Stage Narrative": "Đang lên kế hoạch kể chuyện",
        "Agent Stage Script": "Đang viết kịch bản",
        "Agent Stage Script Revision": "Đang chỉnh sửa kịch bản (vòng phản biện)",
        "Agent Stage Visuals": "Đang lên kế hoạch cảnh & hình ảnh",
        "Agent Stage Titles": "Đang tạo tiêu đề & thumbnail",
        "Agent Stage QA": "Đang kiểm tra chất lượng",
    },
}

ANCHOR = '"Script Generated"'


def main() -> None:
    for lang, values in TRANSLATIONS.items():
        path = I18N_DIR / f"{lang}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        added = 0
        for key in STAGE_KEYS:
            if key in data:
                continue
            data[key] = values[key]
            added += 1
        # Rebuild the file preserving key order: insert stage keys right after
        # the Script Generated anchor so they cluster near the generator UI.
        ordered = {}
        for k, v in data.items():
            ordered[k] = v
            if k == "Script Generated":
                for stage_key in STAGE_KEYS:
                    if stage_key in data and stage_key not in ordered:
                        ordered[stage_key] = data[stage_key]
        text = json.dumps(ordered, ensure_ascii=False, indent=4) + "\n"
        path.write_text(text, encoding="utf-8")
        print(f"{lang}: {added} key(s) added")


if __name__ == "__main__":
    main()
