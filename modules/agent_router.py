def route_query(query: str):
    query_lower = query.lower()
    if "event" in query_lower or "schedule" in query_lower:
        return "event_lookup"
    elif "register" in query_lower or "join" in query_lower:
        return "registration"
    elif "volunteer" in query_lower:
        return "volunteer_info"
    else:
        return "general"
