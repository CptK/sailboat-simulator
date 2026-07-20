# Message type for a waypoint in a sailing route

float32 east  # East coordinate in meters (local frame)
float32 north # North coordinate in meters (local frame)

# True for a waypoint the planner invented to get somewhere — a tacking corner,
# say — which may be skipped once the next real waypoint is within reach.
# False for one that must be visited: a mission via point, a mark to round.
#
# Defaults to false, so a publisher that does not set it asks for every waypoint
# to be honoured, which is the safe reading.
bool is_soft