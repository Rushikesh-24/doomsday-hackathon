# Model benchmark (2026-11-13T11:10:44Z)

Device: `mps` · torch 2.15.0 · macOS-27.0.1-arm64-arm-64bit

Images (10): accessible_toilet_signs.jpg, campus_entrance_rmcad.jpg, corridor_stirling_students.jpg, elevator_lobby_hotel.jpg, ramp_arts.jpg, ramp_hull_museum.jpg, stairs_handrail_sydney.jpg, stairs_oau_campus.jpg, tactile_paving_doncaster.jpg, tactile_paving_singapore.jpg

Latency is warm (first call excluded). Memory figures are process-level and cumulative across models in one run, so treat them as indicative. No ground truth ⇒ no precision/recall reported.

| model | load s | infer ms mean | infer ms p95 | disk MB | detections | classes found |
|---|---|---|---|---|---|---|
| yolo_world_s | 2.78 | 106.4 | 174.0 | 25.9 | 77 | obstacle:28, person:28, signboard:6, entrance_exit:6, elevator:4, ramp:2, walkway:2, staircase:1 |
| yolo_world_l | 2.48 | 134.0 | 183.8 | 94.2 | 71 | obstacle:36, person:25, signboard:6, entrance_exit:2, elevator:1, staircase:1 |
| grounding_dino_tiny | 2.82 | 4400.0 | 4788.6 | 690.3 | 122 | obstacle:33, person:33, elevator:15, tactile_paving:10, handrail:9, entrance_exit:7, walkway:5, signboard:4, ramp:4, staircase:2 |
| owlv2_base | 1.33 | 812.0 | 913.7 | 621.5 | 163 | obstacle:56, handrail:25, signboard:22, person:21, entrance_exit:11, staircase:6, accessibility_symbol:4, door:4, walkway:4, tactile_paving:4, corridor:3, ramp:2, elevator:1 |
| yolo11_coco | 0.09 | 90.5 | 126.8 | 40.7 | 28 | obstacle:14, person:14 |
| campus_ensemble_fast | 0.0 | 153.0 | 191.0 | 134.9 | 38 | obstacle:14, person:14, signboard:6, entrance_exit:2, elevator:1, staircase:1 |
| campus_ensemble | 0.0 | 881.4 | 1024.5 | 662.2 | 128 | obstacle:28, handrail:25, signboard:22, person:14, entrance_exit:11, staircase:6, accessibility_symbol:4, door:4, walkway:4, tactile_paving:4, corridor:3, ramp:2, elevator:1 |

## Per-image top detections

### yolo_world_s
- **accessible_toilet_signs.jpg** (26.6 ms): signboard(information sign) 0.63; signboard(information sign) 0.63; signboard(information sign) 0.47
- **campus_entrance_rmcad.jpg** (140.8 ms): obstacle(bench) 0.83; obstacle(potted plant) 0.80; obstacle(potted plant) 0.73; obstacle(potted plant) 0.25; signboard(signboard) 0.19; entrance_exit(doorway entrance) 0.17; obstacle(chair) 0.15; ramp(accessibility ramp) 0.14
- **corridor_stirling_students.jpg** (141.7 ms): person(person) 0.89; person(person) 0.78; person(person) 0.77; person(person) 0.73; person(person) 0.72; person(person) 0.71; person(person) 0.68; obstacle(chair) 0.66
- **elevator_lobby_hotel.jpg** (193.2 ms): entrance_exit(doorway entrance) 0.21; elevator(elevator door) 0.10
- **ramp_arts.jpg** (41.0 ms): obstacle(chair) 0.21; obstacle(cardboard box) 0.11; person(person) 0.09
- **ramp_hull_museum.jpg** (147.6 ms): obstacle(cardboard box) 0.41; obstacle(cardboard box) 0.40; obstacle(cardboard box) 0.19; entrance_exit(doorway entrance) 0.19; entrance_exit(building entrance) 0.16; obstacle(trash can) 0.16; person(person) 0.14; person(person) 0.09
- **stairs_handrail_sydney.jpg** (145.6 ms): obstacle(bench) 0.31; obstacle(bench) 0.24; obstacle(bench) 0.12; obstacle(bench) 0.12; person(person) 0.08
- **stairs_oau_campus.jpg** (150.5 ms): staircase(staircase) 0.15; entrance_exit(building entrance) 0.12; obstacle(potted plant) 0.08
- **tactile_paving_doncaster.jpg** (41.6 ms): signboard(exit sign) 0.62; walkway(paved path) 0.50; obstacle(trash can) 0.48; person(person) 0.20; obstacle(bicycle) 0.19; person(person) 0.17; person(person) 0.17; person(person) 0.15
- **tactile_paving_singapore.jpg** (35.1 ms): walkway(sidewalk) 0.51

### yolo_world_l
- **accessible_toilet_signs.jpg** (62.7 ms): signboard(information sign) 0.54; signboard(signboard) 0.46; signboard(signboard) 0.36
- **campus_entrance_rmcad.jpg** (158.9 ms): obstacle(bench) 0.88; obstacle(potted plant) 0.84; obstacle(potted plant) 0.75; entrance_exit(doorway entrance) 0.17; signboard(signboard) 0.14; obstacle(potted plant) 0.10
- **corridor_stirling_students.jpg** (176.9 ms): person(person) 0.92; person(person) 0.88; person(person) 0.88; person(person) 0.85; person(person) 0.84; person(person) 0.80; person(person) 0.79; person(person) 0.75
- **elevator_lobby_hotel.jpg** (189.4 ms): signboard(exit sign) 0.28; entrance_exit(doorway entrance) 0.13
- **ramp_arts.jpg** (96.0 ms): obstacle(chair) 0.34; obstacle(chair) 0.16; obstacle(cardboard box) 0.10; elevator(lift door) 0.09
- **ramp_hull_museum.jpg** (167.8 ms): obstacle(trash can) 0.63; signboard(signboard) 0.14; obstacle(bench) 0.13; obstacle(bench) 0.09; obstacle(bench) 0.09
- **stairs_handrail_sydney.jpg** (164.6 ms): obstacle(bench) 0.45; obstacle(bench) 0.40; obstacle(bench) 0.26; obstacle(bench) 0.22; obstacle(bench) 0.21; obstacle(bench) 0.18; obstacle(bench) 0.11; obstacle(bench) 0.10
- **stairs_oau_campus.jpg** (157.6 ms): staircase(flight of steps) 0.27
- **tactile_paving_doncaster.jpg** (87.6 ms): person(person) 0.46; person(person) 0.36; person(person) 0.27; obstacle(trash can) 0.26; obstacle(trash can) 0.22; person(person) 0.18; person(person) 0.16
- **tactile_paving_singapore.jpg** (78.6 ms): obstacle(cardboard box) 0.22

### grounding_dino_tiny
- **accessible_toilet_signs.jpg** (4244.2 ms): obstacle(cardboard box) 0.43; obstacle(cardboard box) 0.42; obstacle(cardboard box) 0.41; signboard(information sign) 0.40; tactile_paving(tactile warning strip) 0.40; person(person) 0.38; tactile_paving(tactile warning strip) 0.37; tactile_paving(tactile warning strip) 0.34
- **campus_entrance_rmcad.jpg** (4221.8 ms): obstacle(potted plant) 0.71; obstacle(potted plant) 0.63; obstacle(bench) 0.62; signboard(directional sign) 0.41; handrail(handrail) 0.40; signboard(exit sign) 0.40; elevator(elevator) 0.38; elevator(elevator) 0.37
- **corridor_stirling_students.jpg** (4891.0 ms): person(person) 0.84; person(person) 0.83; person(person) 0.74; person(person) 0.72; person(person) 0.67; person(person) 0.66; person(person) 0.66; person(person) 0.66
- **elevator_lobby_hotel.jpg** (3541.5 ms): elevator(lift door) 0.46; entrance_exit(doorway entrance) 0.41; tactile_paving(tactile warning strip) 0.40; walkway(footpath) 0.37; elevator(elevator) 0.34; obstacle(barrier) 0.32
- **ramp_arts.jpg** (4135.0 ms): obstacle(chair) 0.57; person(person) 0.50; entrance_exit(doorway entrance) 0.46; obstacle(chair) 0.44; person(person) 0.41; obstacle(trash can) 0.41; elevator(elevator) 0.40; walkway(footpath) 0.39
- **ramp_hull_museum.jpg** (4609.0 ms): obstacle(trash can) 0.48; staircase(staircase) 0.43; obstacle(barrier) 0.41; obstacle(cardboard box) 0.40; walkway(footpath) 0.37; tactile_paving(yellow tactile paving) 0.36; ramp(accessibility ramp) 0.36; person(person) 0.35
- **stairs_handrail_sydney.jpg** (4570.9 ms): handrail(handrail) 0.40; obstacle(barrier) 0.33; person(person) 0.33; tactile_paving(tactile warning strip) 0.30
- **stairs_oau_campus.jpg** (4653.9 ms): person(person) 0.43; obstacle(cardboard box) 0.41; obstacle(potted plant) 0.38; elevator(elevator) 0.37; obstacle(barrier) 0.34; person(person) 0.32; entrance_exit(doorway entrance) 0.32; entrance_exit(doorway entrance) 0.31
- **tactile_paving_doncaster.jpg** (4469.2 ms): person(person) 0.52; obstacle(trash can) 0.52; person(person) 0.51; obstacle(cardboard box) 0.44; obstacle(chair) 0.43; person(person) 0.40; person(person) 0.39; walkway(footpath) 0.38
- **tactile_paving_singapore.jpg** (4663.5 ms): obstacle(cardboard box) 0.57; walkway(footpath) 0.43; signboard(information sign) 0.41; tactile_paving(tactile warning strip) 0.36; handrail(handrail) 0.36; elevator(elevator) 0.31

### owlv2_base
- **accessible_toilet_signs.jpg** (778.2 ms): signboard(information sign) 0.54; accessibility_symbol(accessibility sign) 0.53; accessibility_symbol(accessibility sign) 0.49; accessibility_symbol(wheelchair symbol) 0.47; accessibility_symbol(wheelchair symbol) 0.37; signboard(signboard) 0.27
- **campus_entrance_rmcad.jpg** (779.4 ms): obstacle(bench) 0.63; entrance_exit(doorway entrance) 0.39; door(door) 0.38; staircase(flight of steps) 0.38; signboard(signboard) 0.35; signboard(information sign) 0.32; door(door) 0.30; walkway(sidewalk) 0.27
- **corridor_stirling_students.jpg** (774.6 ms): obstacle(chair) 0.39; obstacle(chair) 0.39; obstacle(bench) 0.34; obstacle(table) 0.34; obstacle(chair) 0.33; obstacle(bench) 0.33; obstacle(suitcase) 0.32; obstacle(table) 0.31
- **elevator_lobby_hotel.jpg** (805.1 ms): elevator(elevator door) 0.61; corridor(hallway) 0.36; door(glass door) 0.27; entrance_exit(doorway entrance) 0.24; door(glass door) 0.23
- **ramp_arts.jpg** (776.5 ms): entrance_exit(doorway entrance) 0.40; staircase(stairs) 0.35; signboard(information sign) 0.35; signboard(exit sign) 0.31; handrail(railing) 0.30; handrail(handrail) 0.29; obstacle(chair) 0.28; signboard(information sign) 0.28
- **ramp_hull_museum.jpg** (786.8 ms): handrail(railing) 0.58; staircase(stairs) 0.45; handrail(railing) 0.39; walkway(walkway) 0.36; handrail(railing) 0.33; obstacle(bench) 0.31; signboard(signboard) 0.27; entrance_exit(building entrance) 0.24
- **stairs_handrail_sydney.jpg** (875.9 ms): staircase(stairs) 0.58; handrail(railing) 0.29; handrail(stair railing) 0.24; handrail(railing) 0.16
- **stairs_oau_campus.jpg** (777.2 ms): staircase(stairs) 0.57; signboard(signboard) 0.23; entrance_exit(building entrance) 0.19; handrail(railing) 0.17; obstacle(trash can) 0.16
- **tactile_paving_doncaster.jpg** (821.1 ms): tactile_paving(yellow tactile paving) 0.41; obstacle(trash can) 0.34; signboard(directional sign) 0.32; signboard(directional sign) 0.29; signboard(information sign) 0.28; signboard(information sign) 0.28; signboard(directional sign) 0.28; obstacle(bench) 0.28
- **tactile_paving_singapore.jpg** (944.7 ms): tactile_paving(tactile paving) 0.54; signboard(information sign) 0.33; tactile_paving(tactile warning strip) 0.17; obstacle(barrier) 0.16

### yolo11_coco
- **accessible_toilet_signs.jpg** (43.2 ms): —
- **campus_entrance_rmcad.jpg** (126.8 ms): obstacle(bench) 0.91; obstacle(potted plant) 0.86; obstacle(potted plant) 0.86
- **corridor_stirling_students.jpg** (111.2 ms): person(person) 0.89; person(person) 0.88; person(person) 0.87; person(person) 0.86; obstacle(chair) 0.85; obstacle(chair) 0.82; person(person) 0.81; person(person) 0.79
- **elevator_lobby_hotel.jpg** (126.7 ms): —
- **ramp_arts.jpg** (57.8 ms): —
- **ramp_hull_museum.jpg** (110.5 ms): —
- **stairs_handrail_sydney.jpg** (111.7 ms): obstacle(bench) 0.57; obstacle(chair) 0.55; obstacle(chair) 0.51; obstacle(chair) 0.47; obstacle(chair) 0.46; obstacle(bench) 0.41; obstacle(couch) 0.37
- **stairs_oau_campus.jpg** (102.5 ms): —
- **tactile_paving_doncaster.jpg** (60.2 ms): person(person) 0.64; person(person) 0.49; person(person) 0.36
- **tactile_paving_singapore.jpg** (54.7 ms): —

### campus_ensemble_fast
- **accessible_toilet_signs.jpg** (104.5 ms): signboard(information sign) 0.54; signboard(signboard) 0.46; signboard(signboard) 0.36
- **campus_entrance_rmcad.jpg** (158.7 ms): obstacle(bench) 0.91; obstacle(potted plant) 0.86; obstacle(potted plant) 0.86; entrance_exit(doorway entrance) 0.17; signboard(signboard) 0.14
- **corridor_stirling_students.jpg** (145.4 ms): person(person) 0.89; person(person) 0.88; person(person) 0.87; person(person) 0.86; obstacle(chair) 0.85; obstacle(chair) 0.82; person(person) 0.81; person(person) 0.79
- **elevator_lobby_hotel.jpg** (212.4 ms): signboard(exit sign) 0.28; entrance_exit(doorway entrance) 0.13
- **ramp_arts.jpg** (164.8 ms): elevator(lift door) 0.09
- **ramp_hull_museum.jpg** (154.4 ms): signboard(signboard) 0.14
- **stairs_handrail_sydney.jpg** (150.5 ms): obstacle(bench) 0.57; obstacle(chair) 0.55; obstacle(chair) 0.51; obstacle(chair) 0.47; obstacle(chair) 0.46; obstacle(bench) 0.41; obstacle(couch) 0.37
- **stairs_oau_campus.jpg** (134.7 ms): staircase(flight of steps) 0.27
- **tactile_paving_doncaster.jpg** (160.2 ms): person(person) 0.64; person(person) 0.49; person(person) 0.36
- **tactile_paving_singapore.jpg** (144.6 ms): —

### campus_ensemble
- **accessible_toilet_signs.jpg** (826.2 ms): signboard(information sign) 0.54; accessibility_symbol(accessibility sign) 0.53; accessibility_symbol(accessibility sign) 0.49; accessibility_symbol(wheelchair symbol) 0.47; accessibility_symbol(wheelchair symbol) 0.37; signboard(signboard) 0.27
- **campus_entrance_rmcad.jpg** (840.1 ms): obstacle(bench) 0.91; obstacle(potted plant) 0.86; obstacle(potted plant) 0.86; entrance_exit(doorway entrance) 0.39; door(door) 0.38; staircase(flight of steps) 0.38; signboard(signboard) 0.35; signboard(information sign) 0.32
- **corridor_stirling_students.jpg** (833.2 ms): person(person) 0.89; person(person) 0.88; person(person) 0.87; person(person) 0.86; obstacle(chair) 0.85; obstacle(chair) 0.82; person(person) 0.81; person(person) 0.79
- **elevator_lobby_hotel.jpg** (868.5 ms): elevator(elevator door) 0.61; corridor(hallway) 0.36; door(glass door) 0.27; entrance_exit(doorway entrance) 0.24; door(glass door) 0.23
- **ramp_arts.jpg** (835.7 ms): entrance_exit(doorway entrance) 0.40; staircase(stairs) 0.35; signboard(information sign) 0.35; signboard(exit sign) 0.31; handrail(railing) 0.30; handrail(handrail) 0.29; signboard(information sign) 0.28; walkway(walkway) 0.28
- **ramp_hull_museum.jpg** (840.7 ms): handrail(railing) 0.58; staircase(stairs) 0.45; handrail(railing) 0.39; walkway(walkway) 0.36; handrail(railing) 0.33; signboard(signboard) 0.27; entrance_exit(building entrance) 0.24; handrail(railing) 0.23
- **stairs_handrail_sydney.jpg** (1031.1 ms): staircase(stairs) 0.58; obstacle(bench) 0.57; obstacle(chair) 0.55; obstacle(chair) 0.51; obstacle(chair) 0.47; obstacle(chair) 0.46; obstacle(bench) 0.41; obstacle(couch) 0.37
- **stairs_oau_campus.jpg** (842.0 ms): staircase(stairs) 0.57; signboard(signboard) 0.23; entrance_exit(building entrance) 0.19; handrail(railing) 0.17; obstacle(trash can) 0.16
- **tactile_paving_doncaster.jpg** (880.4 ms): person(person) 0.64; person(person) 0.49; tactile_paving(yellow tactile paving) 0.41; person(person) 0.36; obstacle(trash can) 0.34; signboard(directional sign) 0.32; signboard(directional sign) 0.29; signboard(information sign) 0.28
- **tactile_paving_singapore.jpg** (1016.5 ms): tactile_paving(tactile paving) 0.54; signboard(information sign) 0.33; tactile_paving(tactile warning strip) 0.17; obstacle(barrier) 0.16
