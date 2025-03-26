#!/bin/bash

# Instalacja Ansible przy użyciu apt
apt update -y  # Aktualizacja repozytoriów przed instalacją
apt install -y ansible

# Weryfikacja instalacji Ansible
echo "Sprawdzam wersję Ansible..."
ansible --version
wersja=$(ansible --version)  # Zapisanie wyjścia do zmiennej
echo "$wersja" > /root/ansible_version.txt  # Zapisanie do pliku z bardziej opisową nazwą

# Wyświetlenie pomocy dla ansible-doc
echo "Wyświetlam pomoc dla ansible-doc:"
ansible-doc --help

# Lista modułów Ansible i zapis do pliku wraz z liczbą
echo "Generowanie listy modułów Ansible i obliczanie ich liczby..."
liczba_modulow=$(ansible-doc -l | wc -l) #Obliczanie liczby i zapisywanie w zmiennej
echo "Liczba dostępnych modułów: $liczba_modulow" > /root/ansible_modules_count.txt
ansible-doc -l > /root/ansible_modules.txt # Zapis wszystkich modułów do pliku

# Wyświetlenie dokumentacji dla modułu setup
echo "Wyświetlam dokumentację dla modułu setup:"
ansible-doc setup

# Wyświetlenie dokumentacji dla modułu copy
echo "Wyświetlam dokumentację dla modułu copy:"
ansible-doc copy

echo "Skrypt zakończony."
