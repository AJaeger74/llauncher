"""Internationalization (i18n) support for llauncher."""

import json
from pathlib import Path


class I18nManager:
    """Thread-safe i18n manager with JSON-based translations."""
    
    _instance = None
    
    def __init__(self):
        if I18nManager._instance is not None:
            raise RuntimeError("I18nManager is a singleton")
        
        self.locales_dir = Path(__file__).parent / "locales"
        self.current_lang = "de"  # Default
        self.translations = {}
        
    def load_language(self, lang_code: str) -> bool:
        """Load translation file for given language code."""
        try:
            lang_file = self.locales_dir / f"{lang_code}.json"
            if not lang_file.exists():
                return False
            
            with open(lang_file, "r", encoding="utf-8") as f:
                self.translations[lang_code] = json.load(f)
            
            # Fallback to English for missing keys
            if lang_code != "en" and "en" in self.translations:
                self._merge_fallback(lang_code, "en")
            
            self.current_lang = lang_code
            return True
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"[i18n] Fehler beim Laden von {lang_code}.json: {e}")
            return False
    
    def _merge_fallback(self, target_lang: str, fallback_lang: str):
        """Merge fallback translations for missing keys."""
        if fallback_lang not in self.translations:
            return
        
        for key, value in self.translations[fallback_lang].items():
            if key not in self.translations[target_lang]:
                self.translations[target_lang][key] = value
    
    def gettext(self, key: str) -> str:
        """Get translated string for key, fallback to English, then key itself."""
        lang_data = self.translations.get(self.current_lang, {})
        
        # Direct lookup in current language
        if key in lang_data:
            return lang_data[key]
        
        # Fallback to English
        en_data = self.translations.get("en", {})
        if key in en_data:
            return en_data[key]
        
        # Ultimate fallback: return the key itself
        print(f"[i18n] Missing translation for '{key}' in {self.current_lang}")
        return key
    
    def get_available_languages(self) -> list[str]:
        """Return list of available language codes."""
        return [f.stem for f in self.locales_dir.glob("*.json")]
    
    def reload(self, lang_code: str = None):
        """Reload translations, optionally switching to new language.
        
        Args:
            lang_code: Language code to switch to (optional)
        """
        if lang_code:
            self.current_lang = lang_code
        # Re-load current language file
        self.load_language(self.current_lang)
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance, creating if necessary."""
        if cls._instance is None:
            cls._instance = I18nManager()
        return cls._instance


# Convenience function for module-level access
def gettext(key: str) -> str:
    """Short-hand for i18n.get_instance().gettext(key)."""
    return I18nManager.get_instance().gettext(key)
