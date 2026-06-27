import asyncio
import time
from collections import Counter

import aiohttp

# ============================
# CONFIGURACIÓN
# ============================
#BASE_URL = "http://iia-base-v1.dsicv.upv.es/"   # pon aquí tu dominio/IP
BASE_URL = "http://mltutor.gti-ia.upv.es/"   # pon aquí tu dominio/IP
VIRTUAL_USERS = 50          # nº de "usuarios" concurrentes
REQUESTS_PER_USER = 20      # peticiones por usuario
CONCURRENCY_LIMIT = 50      # máx peticiones simultáneas
TIMEOUT_SECONDS = 20


async def user_worker(user_id, session, semaphore, results):
    """
    Simula un usuario: mantiene sesión (cookies) y hace varias peticiones
    para comprobar si siempre va al mismo backend.
    """
    upstreams = []
    latencies = []

    for i in range(REQUESTS_PER_USER):
        async with semaphore:
            start = time.perf_counter()
            try:
                async with session.get(BASE_URL, timeout=TIMEOUT_SECONDS) as resp:
                    _ = await resp.text()  # no lo usamos, pero fuerza a leer la respuesta
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)

                    upstream = resp.headers.get("X-Upstream-Addr", "UNKNOWN")
                    upstreams.append(upstream)

            except Exception as e:
                print(f"[user {user_id}] Error en request {i}: {e}")

    results[user_id] = {
        "upstreams": upstreams,
        "latencies": latencies,
    }


async def main():
    connector = aiohttp.TCPConnector(limit=None)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    results = {}

    tasks = []
    sessions = []

    # Un ClientSession por usuario => sesiones separadas (cookies distintas)
    for user_id in range(VIRTUAL_USERS):
        session = aiohttp.ClientSession(connector=connector)
        sessions.append(session)
        task = asyncio.create_task(user_worker(user_id, session, semaphore, results))
        tasks.append(task)

    await asyncio.gather(*tasks)

    for s in sessions:
        await s.close()

    # ============================
    # ANÁLISIS
    # ============================
    all_latencies = []
    upstream_counter = Counter()
    sticky_violations = 0

    for user_id, data in results.items():
        upstreams = data["upstreams"]
        latencies = data["latencies"]

        if not upstreams:
            continue

        all_latencies.extend(latencies)
        upstream_counter.update(upstreams)

        first = upstreams[0]
        if any(u != first for u in upstreams[1:]):
            sticky_violations += 1

    total_requests = sum(len(d["upstreams"]) for d in results.values())

    print("\n========== RESULTADOS ==========\n")
    print(f"URL:                 {BASE_URL}")
    print(f"Usuarios virtuales:  {VIRTUAL_USERS}")
    print(f"Requests/usuario:    {REQUESTS_PER_USER}")
    print(f"Total requests:      {total_requests}")

    if all_latencies:
        all_latencies.sort()
        avg = sum(all_latencies) / len(all_latencies)
        p50 = all_latencies[int(0.50 * len(all_latencies))]
        p90 = all_latencies[int(0.90 * len(all_latencies))]
        p99 = all_latencies[int(0.99 * len(all_latencies))]

        print(f"\nLatencia media:      {avg*1000:.1f} ms")
        print(f"P50:                 {p50*1000:.1f} ms")
        print(f"P90:                 {p90*1000:.1f} ms")
        print(f"P99:                 {p99*1000:.1f} ms")
    else:
        print("\nNo hay latencias registradas.")

    print("\nReparto de peticiones por instancia (escala):")
    for up, count in upstream_counter.most_common():
        print(f"  {up}: {count} requests")

    print(f"\nUsuarios con violaciones de sticky session: {sticky_violations} / {VIRTUAL_USERS}")
    if sticky_violations == 0:
        print("✅ Sticky sessions OK: cada usuario ha ido siempre al mismo backend.")
    else:
        print("⚠️ Hay usuarios que han saltado de backend. Revisa la config de sticky en Nginx.")


if __name__ == "__main__":
    asyncio.run(main())
