---
type: Тезис
сверка: не сверен
---

# Показатели PWA считаются во временной области

## Формулировка

Все ходовые индикаторы pulse wave analysis (PWA) вычисляются во временной области по исходной волне и двум-трём её первым производным. Частотная, время-частотная, time-scale области, blind source separation и нелинейный анализ применялись в единичных работах.

## Следствие для проекта

Делинеация (начало волны, пик) — не вспомогательная операция, а вход всех индексов. Ошибка положения пика в 20 мс и начала волны в 30 мс искажает производные сильнее, чем саму волну. Поэтому приёмка модулятора (`[[checks/check_1.py]]`) нормирует время пика медианой 15 мс и p99 50 мс, а не «на глаз по рисунку».

## Связи

- тип: [[Тезис]]
- утверждаетсяВ: [[Charlton 2023]]

## Цитата

- Страница: PDF с. 22 (колонтитул 21), раздел 7 «Pulse wave analysis», подраздел Current and future challenges.
- Текст: «Time domain-based algorithms using joint analysis of original, and two or three first derivatives of the PPG waveforms were offered first (Charlton et al 2022). All popular PWA indicators are calculated in the time domain.»
