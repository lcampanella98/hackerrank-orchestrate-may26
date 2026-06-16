from config import SUPPORT_TICKETS_DIR
from graph.graph import app
from program_io import read_tickets, write_tickets
import asyncio
from typing import List, Any

SAMPLE = False
RUN_PARALLEL = False
MAX_CONCURRENCY = 5

if not SAMPLE:
    input_path = SUPPORT_TICKETS_DIR / "support_tickets.csv"
    output_path = SUPPORT_TICKETS_DIR / "output.csv"
else:
    input_path = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"
    output_path = SUPPORT_TICKETS_DIR / "output_test.csv"



async def process_ticket_async(ticket) -> Any:
    print(f"processing ticket {ticket}")
    result = await app.ainvoke(ticket)
    print(f"processed ticket {ticket}")
    return result


def process_ticket_sync(ticket) -> Any:
    print(f"processing ticket {ticket}")
    result = app.invoke(ticket)
    print(f"processed ticket {ticket}")
    return result


async def run_parallel(tickets: List[Any]) -> List[Any]:
    if MAX_CONCURRENCY is None:
        tasks = [process_ticket_async(t) for t in tickets]
        return await asyncio.gather(*tasks)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def sem_task(ticket):
        async with semaphore:
            return await process_ticket_async(ticket)

    tasks = [sem_task(t) for t in tickets]
    return await asyncio.gather(*tasks)


async def run_sequential(tickets: List[Any]) -> List[Any]:
    results = []
    for ticket in tickets:
        result = await process_ticket_async(ticket)
        results.append(result)
    return results


# ---- Entry point ----
def main():
    tickets = read_tickets(input_path)

    if RUN_PARALLEL:
        results = asyncio.run(run_parallel(tickets))
    else:
        results = asyncio.run(run_sequential(tickets))

    write_tickets(output_path, results)


if __name__ == "__main__":
    main()