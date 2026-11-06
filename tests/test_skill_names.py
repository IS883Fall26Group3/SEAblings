"""Skill-name cleaning and the alias table's build rules. No downloads needed: small tables inline."""
import sys
from pathlib import Path

from core.skill_names import normalize, skill_key, standard_name

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_skill_aliases import acronym_variants, esco_alternatives, layer, onet_variants  # noqa: E402

TABLE = {"phyton": "Python", "ms excel": "Excel", "microsoft excel": "Excel", "contract management": "manage contracts"}


def test_normalize_keeps_programming_names():
    assert normalize("  C++ / C# & ASP.NET ") == "c++ c# and asp.net"
    assert normalize("Excel (VBA).") == "excel vba"


def test_normalize_us_spelling_and_skills_suffix():
    assert normalize("Data Visualisation") == "data visualization"
    assert normalize("Financial Modelling") == "financial modeling"
    assert normalize("Organisational Behaviour") == "organizational behavior"
    assert normalize("Expertise in Advertising") == "expertise in advertising"  # not -ize words
    assert normalize("Communication Skills") == "communication"
    assert normalize("Soft Skills") == "soft skills" and normalize("Skills") == "skills"


def test_standard_names_and_keys():
    assert standard_name("Phyton", TABLE) == "Python"
    assert skill_key("MS Excel", TABLE) == skill_key("Microsoft Excel", TABLE) == "excel"
    assert standard_name("LPG Distribution", TABLE) == "LPG Distribution"  # unknown: only cleaned for the key
    assert skill_key("Contract Management", TABLE) == "manage contracts"


def test_onet_variants():
    assert acronym_variants("Structured query language SQL") == ["Structured query language", "SQL"]
    assert "LT" not in acronym_variants("Verity Software House ModFit LT")  # not the initials before it
    assert "Power BI" in onet_variants("Microsoft Power BI")
    assert "Access" not in onet_variants("Microsoft Access")  # an everyday word alone
    assert "Business Intelligence" not in onet_variants("SAP Business Intelligence")  # a general idea
    assert "Digital Analytics" not in onet_variants("IBM Digital Analytics")


def test_esco_alternatives_keep_wordings_not_neighbours():
    alts = ["managing contracts", "contract management", "contract law", "data analysis"]
    assert esco_alternatives("manage contracts", alts) == ["managing contracts", "contract management"]
    assert esco_alternatives("statistics", ["data analysis", "statistical analysis", "statistic"]) == ["statistic"]
    assert esco_alternatives("think analytically", ["analytical thinking"]) == ["analytical thinking"]
    assert esco_alternatives("Microsoft Access", ["Access", "MS Access"]) == ["MS Access"]


def test_layer_drops_variants_claimed_twice():
    d = layer([("ASP", "Microsoft Active Server Pages ASP"), ("ASP", "Other ASP"), ("Python", "Python")])
    assert "asp" not in d and d["python"] == "Python"
