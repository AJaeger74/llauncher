

def _extracted_load_running_process_args():
    # NOTE: Extracted from llauncher.py lines 333-468
    # To restore, remove the thin delegate and uncomment the body
    pass


    def load_running_process_args(self, show_dialogs: bool = True):
        """Liest Parameter des laufenden llama-server und lädt sie in die App.
        
        Args:
            show_dialogs: Ob Warnungen/Dialoge angezeigt werden sollen (default: True)
        """
        from process_runner import read_and_apply_running_args
        
        # Prüfen ob UI-Komponenten initialisiert sind (Schutz vor Race Condition)
        if not hasattr(self, 'param_sliders') or not self.param_sliders:
            return False
        
        # Zuerst check_existing_process() aufrufen, um external_runner_args zu setzen
        self.check_existing_process()
        
        # UI Components Dict zusammenstellen – greift auf param_sliders zu
        ui_components = {
            'model_line': self.model_line if hasattr(self, 'model_line') else None,
            'mmproj_line': self.mmproj_line if hasattr(self, 'mmproj_line') else None,
            # Integer-Slider (param_sliders[key] = {"slider": DirectClickSlider, "edit": QLineEdit})
            'ctx_slider': self.param_sliders.get('-c', {}).get('slider'),
            'ctx_edit': self.param_sliders.get('-c', {}).get('edit'),
            'batch_slider': self.param_sliders.get('-b', {}).get('slider'),
            'batch_edit': self.param_sliders.get('-b', {}).get('edit'),
            'threads_slider': self.param_sliders.get('-t', {}).get('slider'),
            'threads_edit': self.param_sliders.get('-t', {}).get('edit'),
            'gpu_layers_slider': self.param_sliders.get('-ngl', {}).get('slider'),
            'gpu_layers_edit': self.param_sliders.get('-ngl', {}).get('edit'),
            # -ngl "all" Checkbox (wenn vorhanden)
            'ngl_all_checkbox': getattr(self, 'ngl_all_checkbox', None),
            'parallel_slider': self.param_sliders.get('-np', {}).get('slider'),
            'parallel_edit': self.param_sliders.get('-np', {}).get('edit'),
            # Float-Slider (gleiche Struktur)
            'temp_slider': self.param_sliders.get('--temp', {}).get('slider'),
            'temp_edit': self.param_sliders.get('--temp', {}).get('edit'),
            'top_p_slider': self.param_sliders.get('--top-p', {}).get('slider'),
            'top_p_edit': self.param_sliders.get('--top-p', {}).get('edit'),
            'repeat_penalty_slider': self.param_sliders.get('--repeat-penalty', {}).get('slider'),
            'repeat_penalty_edit': self.param_sliders.get('--repeat-penalty', {}).get('edit'),
            # ComboBox (param_sliders[key] = {"combo": QComboBox})
            'cache_type_k_combo': self.param_sliders.get('--cache-type-k', {}).get('combo'),
            'cache_type_v_combo': self.param_sliders.get('--cache-type-v', {}).get('combo'),
            'flash_attn_combo': self.param_sliders.get('--flash-attn', {}).get('combo'),
            # Text Input (param_sliders[key] = {"edit": QLineEdit})
            'host_edit': self.param_sliders.get('--host', {}).get('edit'),
            'save_path_edit': self.param_sliders.get('--slot-save-path', {}).get('edit'),
        }
        
         # Flag setzen um zu verhindern dass on_model_selected() den Slider überschreibt
        self.loading_running_args = True
        
        external_args, model_path, exe_path, pid_found = read_and_apply_running_args(
            self
        )
        
        if not pid_found:
            self.loading_running_args = False  # Reset auch im Fehlerfall
            if show_dialogs:
                QMessageBox.warning(self, translatable("msg_no_running_process_title"),
                                  translatable("msg_no_llama_server_found"))
            return False
        
         # Modell-Pfad aufteilen in Verzeichnis + Dateiname
        if model_path:
            import os
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path)
            
            # Setze "Modelle"-Feld (Verzeichnis) und internes Attribut
            if hasattr(self, 'model_line'):
                self.model_line.setText(model_dir)
            
            # Modell-Verzeichnis aktualisieren und ComboBox neu laden
            if hasattr(self, 'model_directory'):
                self.model_directory = model_dir
 
            
            # ComboBox neu füllen mit Modellen aus neuem Verzeichnis
            if hasattr(self, 'model_combo'):
                self.update_model_dropdown()
                
                # Jetzt nach model_name im UserRole suchen
                found_index = -1
                for i in range(self.model_combo.count()):
                    user_data = self.model_combo.itemData(i, role=Qt.ItemDataRole.UserRole)
                    if user_data and user_data == model_name:
                        found_index = i
                        break
                
                if found_index >= 0:
                    self.model_combo.setCurrentIndex(found_index)
                else:
                    # Falls Datei nicht in ComboBox ist, direkt den Dateinamen setzen
                    self.model_combo.setCurrentText(model_name)
                
                # Internes selected_model setzen (wird von get_current_args() benötigt)
                self.selected_model = model_path
        
         # Exe-Pfad setzen (falls vorhanden) - nur Verzeichnis ohne Filename
        if exe_path and hasattr(self, 'exe_line'):
            import os
            norm = exe_path
            while norm.endswith('/llama-server'):
                norm = os.path.dirname(norm)
            if 'build/bin' in norm:
                norm = norm.split('build/bin')[0].rstrip('/')
            if 'build/' in norm:
                norm = norm.split('build/')[0].rstrip('/')
            self.exe_line.setText(norm)
            # llama_cpp_path aktualisieren, damit nächster Start das richtige Binary lädt
            if hasattr(self, 'llama_cpp_path'):
                self.llama_cpp_path = norm
                save_config({"llama_cpp_path": norm})
        
     # Externe Parameter anzeigen (nicht in APP verwaltet) – nur wenn es welche gibt
        if external_args and len(external_args) > 0:
            # Populiere das Custom Commands Feld anstelle eines Dialogs
            self._format_external_args_to_text(external_args)
        
        # Externe Parameter speichern für Debug-Ausgabe und Kommandozeilen-Generierung
        self.external_args = external_args
        
        # Debug-Bereich aktualisieren mit vollständiger Kommandozeile (inkl. externer Args)
        try:
            command = self.build_full_command()
            self.debug_text.setText(command)
        except Exception as e:
            self.debug_text.append(f"⚠️ Konnte Debug-Ausgabe nicht aktualisieren: {e}")
        
        if external_args is not None:
            self.debug_text.append(translatable("msg_loaded_params", pid=pid_found, count=len(external_args)))
        
        # Flag zurücksetzen – jetzt darf on_model_selected() wieder normal arbeiten
        self.loading_running_args = False

