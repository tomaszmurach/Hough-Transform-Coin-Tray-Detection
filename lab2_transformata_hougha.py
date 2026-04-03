import cv2
import numpy as np

# Awaryjny prog podzialu nominalow.
# Jest uzywany tylko wtedy, gdy nie da sie wyznaczyc progu adaptacyjnie.
RADIUS_THRESHOLD = 34


def process_image(image, image_name):
    """
    Przetwarza pojedynczy obraz i zwraca komplet wynikow.

    Etapy:
    1. Wstepne przetwarzanie obrazu.
    2. Wykrycie tacki.
    3. Wykrycie monet.
    4. Klasyfikacja polozenia monet wzgledem tacki.
    5. Klasyfikacja nominalow.
    6. Zliczenie monet i sum wartosci.
    7. Przygotowanie wizualizacji.
    """
    preprocessed = preprocess_image(image)

    tray_rect = detect_tray(preprocessed)
    coins = detect_coins(preprocessed, tray_rect)

    classified_coins = classify_coins_position(coins, tray_rect)
    valued_coins = classify_coin_nominals(classified_coins)

    stats = count_and_sum(valued_coins)
    visualization = draw_results(image, tray_rect, valued_coins, stats, image_name)

    return {
        "tray_rect": tray_rect,
        "coins": valued_coins,
        "stats": stats,
        "visualization": visualization
    }


def preprocess_image(image):
    """
    Przygotowuje obraz do dalszej analizy.

    Tworzone sa dwie wersje danych:
    - do wykrywania monet,
    - do wykrywania tacki.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Poprawa lokalnego kontrastu.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Wersja do wykrywania monet.
    coins_blurred = cv2.GaussianBlur(equalized, (7, 7), 1.5)

    # Wersja do wykrywania tacki.
    tray_blurred = cv2.GaussianBlur(equalized, (5, 5), 1.0)
    tray_edges = cv2.Canny(tray_blurred, 50, 140)

    # Domkniecie drobnych przerw w krawedziach.
    close_kernel = np.ones((3, 3), np.uint8)
    tray_edges = cv2.morphologyEx(tray_edges, cv2.MORPH_CLOSE, close_kernel)

    # Dodatkowe domkniecie w pionie, zeby lepiej polaczyc boki tacki.
    vertical_kernel = np.ones((11, 1), np.uint8)
    tray_edges = cv2.morphologyEx(tray_edges, cv2.MORPH_CLOSE, vertical_kernel)

    return {
        "gray": gray,
        "equalized": equalized,
        "blurred": coins_blurred,
        "edges": tray_edges
    }


def detect_tray(preprocessed):
    """
    Wykrywa tacke z wykorzystaniem transformacji Hougha dla linii.

    Podejscie:
    1. Wykrycie pionowych linii i wybor lewego oraz prawego boku tacki.
    2. Ograniczenie obszaru analizy na podstawie tych bokow.
    3. Wyznaczenie gornej i dolnej krawedzi z linii poziomych.
    4. Uzycie fallbacku opartego na odcinkach pionowych, jesli poziome
       krawedzie nie zostana wykryte wystarczajaco dobrze.
    """
    edges = preprocessed["edges"]
    h, w = edges.shape

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=140,
        maxLineGap=40
    )

    if lines is None:
        raise ValueError("detect_tray: HoughLinesP nie wykryl zadnych linii.")

    vertical_candidates = []
    horizontal_candidates = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = int(np.hypot(x2 - x1, y2 - y1))
        x_avg = (x1 + x2) // 2
        y_avg = (y1 + y2) // 2

        # Kandydaci na pionowe boki tacki.
        if dx < 35 and dy > 140:
            vertical_candidates.append({
                "x": x_avg,
                "y1": min(y1, y2),
                "y2": max(y1, y2),
                "length": length
            })

        # Kandydaci na poziome krawedzie tacki.
        elif dy < 25 and dx > 80:
            horizontal_candidates.append({
                "x1": min(x1, x2),
                "x2": max(x1, x2),
                "y": y_avg,
                "length": length
            })

    if len(vertical_candidates) < 2:
        raise ValueError("detect_tray: za malo linii pionowych do wyznaczenia tacki.")

    # Grupowanie pionowych linii lezacych blisko siebie.
    vertical_candidates.sort(key=lambda c: c["x"])
    clusters = []
    current_cluster = [vertical_candidates[0]]
    tolerance = 25

    for candidate in vertical_candidates[1:]:
        if abs(candidate["x"] - current_cluster[-1]["x"]) <= tolerance:
            current_cluster.append(candidate)
        else:
            clusters.append(current_cluster)
            current_cluster = [candidate]

    clusters.append(current_cluster)

    vertical_clusters = []
    for cluster in clusters:
        total_weight = sum(item["length"] for item in cluster)
        mean_x = int(sum(item["x"] * item["length"] for item in cluster) / total_weight)
        min_y = min(item["y1"] for item in cluster)
        max_y = max(item["y2"] for item in cluster)

        vertical_clusters.append({
            "x": mean_x,
            "top": min_y,
            "bottom": max_y,
            "weight": total_weight
        })

    if len(vertical_clusters) < 2:
        raise ValueError("detect_tray: za malo klastrow pionowych po scaleniu.")

    # Wybor pary pionowych bokow najlepiej odpowiadajacej szerokosci tacki.
    best_pair = None
    best_score = float("inf")

    for i in range(len(vertical_clusters)):
        for j in range(i + 1, len(vertical_clusters)):
            x1 = vertical_clusters[i]["x"]
            x2 = vertical_clusters[j]["x"]

            left_x = min(x1, x2)
            right_x = max(x1, x2)
            width = right_x - left_x

            if width < 180 or width > 340:
                continue

            weight_sum = vertical_clusters[i]["weight"] + vertical_clusters[j]["weight"]
            expected_width = 260
            score = abs(width - expected_width) - 0.01 * weight_sum

            if score < best_score:
                best_score = score
                best_pair = (vertical_clusters[i], vertical_clusters[j])

    if best_pair is None:
        raise ValueError("detect_tray: nie znaleziono sensownej pary pionowych bokow tacki.")

    left_cluster, right_cluster = sorted(best_pair, key=lambda c: c["x"])

    left = left_cluster["x"]
    right = right_cluster["x"]

    if left >= right:
        raise ValueError("detect_tray: niepoprawne granice pionowe tacki.")

    # Wybiera poziome linie, ktore realnie przecinaja obszar tacki.
    tray_width = right - left
    horizontal_roi = []

    for candidate in horizontal_candidates:
        overlap_left = max(candidate["x1"], left - 15)
        overlap_right = min(candidate["x2"], right + 15)
        overlap = overlap_right - overlap_left

        if overlap > 0.55 * tray_width:
            horizontal_roi.append(candidate)

    top = None
    bottom = None

    if len(horizontal_roi) >= 2:
        top_candidates = [c for c in horizontal_roi if c["y"] < h * 0.45]
        bottom_candidates = [c for c in horizontal_roi if c["y"] > h * 0.45]

        if top_candidates:
            top = min(c["y"] for c in top_candidates)

        if bottom_candidates:
            bottom = max(c["y"] for c in bottom_candidates)

    # Fallback dla gornej i dolnej krawedzi, gdy poziome linie sa niewystarczajace.
    if top is None or bottom is None:
        inner_verticals = [
            candidate for candidate in vertical_candidates
            if left - 25 <= candidate["x"] <= right + 25
        ]

        if len(inner_verticals) < 2:
            raise ValueError("detect_tray: za malo pionowych odcinkow wewnatrz tacki.")

        y_starts = sorted(candidate["y1"] for candidate in inner_verticals)
        y_ends = sorted(candidate["y2"] for candidate in inner_verticals)

        if len(y_starts) >= 3:
            inner_top = y_starts[1]
        else:
            inner_top = y_starts[0]

        inner_bottom = y_ends[-1]

        side_top = max(left_cluster["top"], right_cluster["top"])
        side_bottom = max(left_cluster["bottom"], right_cluster["bottom"])

        if top is None:
            top = max(inner_top, side_top)
        if bottom is None:
            bottom = max(inner_bottom, side_bottom)

    if top >= bottom:
        raise ValueError("detect_tray: niepoprawne granice poziome tacki.")

    # Margines poprawia stabilnosc klasyfikacji monet przy samej krawedzi tacki.
    margin_x = 10
    margin_y = 18

    tray_rect = {
        "left": max(0, left - margin_x),
        "right": min(w - 1, right + margin_x),
        "top": max(0, int(top) - margin_y),
        "bottom": min(h - 1, int(bottom) + margin_y)
    }

    return tray_rect


def detect_coins(preprocessed, tray_rect):
    """
    Wykrywa monety za pomoca transformacji Hougha dla okregow.

    Po wykryciu okregow odrzucane sa falszywe detekcje, ktore pojawiaja sie
    w zaokraglonych rogach tacki.
    """
    blurred = preprocessed["blurred"]

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=25,
        minRadius=12,
        maxRadius=40
    )

    coins = []

    if circles is None:
        return coins

    circles = np.round(circles[0, :]).astype("int")

    left = tray_rect["left"]
    right = tray_rect["right"]
    top = tray_rect["top"]
    bottom = tray_rect["bottom"]

    # Strefy rogow tacki, w ktorych odrzucane sa falszywe okregi.
    corner_margin_x = 75
    corner_margin_y = 75

    for (x, y, r) in circles:
        on_tray = left <= x <= right and top <= y <= bottom

        if on_tray:
            in_top_left_corner = (
                    left <= x <= left + corner_margin_x and
                    top <= y <= top + corner_margin_y
            )

            in_top_right_corner = (
                    right - corner_margin_x <= x <= right and
                    top <= y <= top + corner_margin_y
            )

            in_bottom_left_corner = (
                    left <= x <= left + corner_margin_x and
                    bottom - corner_margin_y <= y <= bottom
            )

            in_bottom_right_corner = (
                    right - corner_margin_x <= x <= right and
                    bottom - corner_margin_y <= y <= bottom
            )

            fake_corner_circle = (
                    in_top_left_corner or
                    in_top_right_corner or
                    in_bottom_left_corner or
                    in_bottom_right_corner
            )

            if fake_corner_circle:
                continue

        coins.append({"x": x, "y": y, "r": r})

    return coins


def classify_coins_position(coins, tray_rect):
    """
    Klasyfikuje monety na podstawie polozenia ich srodka.

    Jesli srodek monety znajduje sie wewnatrz prostokata tacki,
    moneta jest traktowana jako lezaca na tacy.
    """
    result = []

    for coin in coins:
        x, y = coin["x"], coin["y"]

        on_tray = (
                tray_rect["left"] <= x <= tray_rect["right"] and
                tray_rect["top"] <= y <= tray_rect["bottom"]
        )

        coin["location"] = "on_tray" if on_tray else "outside_tray"
        result.append(coin)

    return result


def classify_coin_nominals(coins):
    """
    Klasyfikuje nominal monety na podstawie promienia wykrytego okregu.

    Prog podzialu jest wyznaczany adaptacyjnie na podstawie najwiekszej
    przerwy pomiedzy posortowanymi promieniami. Gdy nie jest to mozliwe,
    uzywany jest prog awaryjny.
    """
    if not coins:
        return coins

    radii = sorted(coin["r"] for coin in coins)

    if len(radii) < 2:
        threshold = RADIUS_THRESHOLD
    else:
        max_gap = -1
        threshold = RADIUS_THRESHOLD

        for i in range(len(radii) - 1):
            gap = radii[i + 1] - radii[i]

            if gap > max_gap:
                max_gap = gap
                threshold = (radii[i] + radii[i + 1]) / 2.0

    for coin in coins:
        r = coin["r"]

        if r >= threshold:
            coin["nominal"] = "5zl"
            coin["value"] = 5.00
        else:
            coin["nominal"] = "5gr"
            coin["value"] = 0.05

    return coins


def count_and_sum(coins):
    """
    Zlicza monety i sumuje ich wartosci w dwoch grupach:
    - on_tray,
    - outside_tray.
    """
    stats = {
        "on_tray": {"count": 0, "value": 0.0},
        "outside_tray": {"count": 0, "value": 0.0}
    }

    for coin in coins:
        loc = coin["location"]
        stats[loc]["count"] += 1
        stats[loc]["value"] += coin["value"]

    return stats


def draw_results(image, tray_rect, coins, stats, image_name):
    """
    Rysuje wizualizacje koncowa na kopii obrazu.

    Na obrazie nanoszone sa:
    - prostokat tacki,
    - wykryte monety,
    - podpis nominalu przy kazdej monecie,
    - podsumowanie liczby i wartosci monet,
    - nazwa przetwarzanego obrazu.
    """
    output = image.copy()

    if tray_rect is not None and all(key in tray_rect for key in ["left", "right", "top", "bottom"]):
        cv2.rectangle(
            output,
            (tray_rect["left"], tray_rect["top"]),
            (tray_rect["right"], tray_rect["bottom"]),
            (0, 255, 255),
            2
        )

    for coin in coins:
        color = (0, 255, 0) if coin["location"] == "on_tray" else (0, 0, 255)

        cv2.circle(output, (coin["x"], coin["y"]), coin["r"], color, 2)

        cv2.putText(
            output,
            coin["nominal"],
            (coin["x"] - 20, coin["y"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.putText(
        output,
        f"Obraz: {image_name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Na tacy: {stats['on_tray']['count']} szt., {stats['on_tray']['value']:.2f} zl",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Poza taca: {stats['outside_tray']['count']} szt., {stats['outside_tray']['value']:.2f} zl",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return output


def print_summary(result):
    """
    Wypisuje krotkie podsumowanie wynikow do konsoli.
    """
    stats = result["stats"]

    print("\n" + "=" * 40)
    print("Podsumowanie obrazu")
    print(f"Na tacy     -> liczba monet: {stats['on_tray']['count']}, suma: {stats['on_tray']['value']:.2f} zl")
    print(
        f"Poza taca   -> liczba monet: {stats['outside_tray']['count']}, suma: {stats['outside_tray']['value']:.2f} zl")


def main():
    """
    Funkcja glowna programu.

    Program przetwarza obrazy tray1.jpg ... tray8.jpg,
    wyswietla wynik dla kazdego obrazu i drukuje podsumowanie.
    """
    image_paths = [f"tray{i}.jpg" for i in range(1, 9)]

    for path in image_paths:
        image = cv2.imread(path)

        if image is None:
            print(f"Nie udalo sie wczytac obrazu: {path}")
            continue

        try:
            result = process_image(image, path)
        except ValueError as e:
            print(f"\nBlad dla obrazu {path}: {e}")
            continue

        cv2.imshow(path, result["visualization"])
        print_summary(result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
