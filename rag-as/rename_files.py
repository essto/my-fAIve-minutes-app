import os

def rename_files_replace_spaces(directory_path):
    """
    Zmienia nazwy wszystkich plików w podanym katalogu, 
    zamieniając spacje na znaki podkreślenia '_'.
    
    Args:
        directory_path: Ścieżka do katalogu z plikami
    """
    # Sprawdź czy podany katalog istnieje
    if not os.path.isdir(directory_path):
        print(f"Błąd: Katalog '{directory_path}' nie istnieje.")
        return
    
    # Liczniki dla statystyk
    total_files = 0
    renamed_files = 0
    
    # Przejdź przez wszystkie pliki w katalogu
    for filename in os.listdir(directory_path):
        # Pomiń katalogi, zajmij się tylko plikami
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            total_files += 1
            
            # Sprawdź czy nazwa pliku zawiera spacje
            if ' ' in filename:
                # Utwórz nową nazwę zamieniając spacje na podkreślenia
                new_filename = filename.replace(' ', '_')
                new_file_path = os.path.join(directory_path, new_filename)
                
                try:
                    # Zmień nazwę pliku
                    os.rename(file_path, new_file_path)
                    print(f"Zmieniono: '{filename}' -> '{new_filename}'")
                    renamed_files += 1
                except Exception as e:
                    print(f"Błąd podczas zmiany nazwy '{filename}': {str(e)}")
    
    # Wyświetl podsumowanie
    print(f"\nPodsumowanie:")
    print(f"Przeszukany katalog: {directory_path}")
    print(f"Znaleziono plików: {total_files}")
    print(f"Zmieniono nazw plików: {renamed_files}")


if __name__ == "__main__":
    # Można zmienić katalog wejściowy, domyślnie używamy "./data"
    directory = "./data"
    
    print(f"Rozpoczynam zmianę nazw plików w katalogu: {directory}")
    rename_files_replace_spaces(directory)
