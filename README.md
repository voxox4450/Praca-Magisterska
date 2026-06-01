# Kinodynamic UAV Path Planning Simulator (Risk-Aware A*)

Symulator lotu bezzałogowych statków powietrznych (UAV) realizujący oraz porównujący klasyczne algorytmy poszukiwania ścieżki z autorskim, kinodynamicznym podejściem uwzględniającym masę i bezwładność platformy. Projekt stanowi integralną część pracy magisterskiej.

---

## Główne funkcjonalności
* **Implementacja algorytmów:** Dijkstra, Standardowy A* oraz autorski **Risk-Aware A***.
* **Model kinodynamiczny:** Uwzględnienie masy drona ($m$), siły ciągu netto, promienia skrętu oraz dynamicznie obliczanej bezpiecznej prędkości manewru.
* **Symulacja dynamiczna (Online):** Replanowanie trasy w scenariuszu nagłego pojawienia się przeszkód w zasięgu sensorów platformy.
* **Weryfikacja kryterium wykonalności:** Analiza dystansu hamowania, czasu reakcji oraz buforów bezpieczeństwa dla różnych mas (od 1 kg do 50 kg).

---

## 📂 Struktura projektu

```text
├── algorithms/          # Implementacje algorytmów (Dijkstra, A*, Risk-Aware A*)
│   ├── common.py        # Funkcje pomocnicze, analizy ryzyka i czasu lotu
│   ├── dijkstra.py      # Klasyczny algorytm Dijkstry
│   ├── a_star.py        # Standardowy algorytm A*
│   └── a_star_risk.py   # Kinodynamiczny algorytm Risk-Aware A*
├── environment/         # Definicja i generowanie środowiska testowego
│   └── grid_map.py      # Klasa GridMap (reprezentacja mapy i stref ryzyka)
├── visualization/       # Narzędzia do renderowania wykresów i symulacji
│   ├── plotter.py       # Interaktywny symulator online ze sliderami
│   └── metrics_terminal.py # Porównawcze raporty hamowania w konsoli
├── config.py            # Globalna konfiguracja fizyczna i algorytmiczna
├── requirements.txt     # Zależności biblioteczne projektu
└── main.py              # Główny skrypt uruchomieniowy symulacji
