# CEREBRO Security & Privacy

## Purpose

CEREBRO includes defense-in-depth controls around audio-processing and profile features. The goal is to make the system safer to run locally and easier to harden before deployment.

## Data handling

The profile services store compact JSON summaries and explicitly supplied personalization feedback. They do not persist raw microphone recordings or raw ASR transcripts.

Persistent profile data is stored under:

- `PERSONAL_BASELINE_DIR`
- `VOICE_HISTORY_DIR`
- `PERSONALIZATION_DIR`

Use a storage path outside the source tree in production.

## Retention

Automatic retention pruning is enabled by default for 30 days.

```env
PRIVACY_RETENTION_ENABLED=true
PRIVACY_RETENTION_DAYS=30
```

Pruning runs during application initialization. The retention clock follows file modification time because the profile stores are updated when data is written.

## One-click profile deletion

```text
DELETE /api/v1/privacy/{profile_id}
```

This deletes the profile's baseline, voice-history and personalization files.

The browser still retains its anonymous profile ID in local storage unless the user clears it separately. Reusing the same ID after deletion starts with an empty server-side profile.

## Profile ID hardening

Profile IDs are restricted to 1–64 ASCII-safe alphanumeric/underscore/hyphen characters. They are never directly used as an unchecked filesystem path component.

## HTTP security middleware

The backend middleware provides:

- `X-Request-ID`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: no-referrer`
- `Permissions-Policy` with microphone restricted to the current origin
- `Cache-Control: no-store` for API responses
- global body-size defense in depth through `Content-Length`
- per-client/per-route sliding-window rate limiting
- optional API-key authentication
- optional HSTS header

The global body-size check is deliberately defense-in-depth. Individual audio endpoints retain their own stricter upload and duration checks.

## API-key authentication

Enable with:

```env
API_AUTH_ENABLED=true
API_AUTH_KEY=<long-random-secret>
```

The frontend can send the key through:

```env
VITE_CEREBRO_API_KEY=<same-secret>
```

Health/root endpoints remain available without the key so deployments can perform basic liveness checks. CORS preflight requests are not forced through the API-key gate.

The API key is intended for a controlled deployment or trusted front-end. It is not a replacement for a full identity/authorization system.

## WebSocket security

Live WebSocket endpoints perform:

- configured-origin checking when a browser supplies an `Origin` header
- per-IP concurrent connection limiting
- optional authentication using a token supplied in the initial `start` control message
- existing chunk-size limits
- existing maximum session-duration limits

Enable token authentication with:

```env
WEBSOCKET_AUTH_ENABLED=true
WEBSOCKET_AUTH_TOKEN=<long-random-secret>
```

The browser may provide:

```env
VITE_CEREBRO_WS_TOKEN=<same-secret>
```

The token is sent in the JSON `start` message rather than a URL query parameter so the secret does not become part of the WebSocket URL.

## CORS

Production deployments should replace the localhost defaults with the exact HTTPS origin of the deployed frontend.

```env
ALLOWED_ORIGINS=https://your-frontend.example
```

Do not use `*` with credentials-enabled CORS.

## HTTPS

Set:

```env
SECURITY_HSTS_ENABLED=true
```

only when the application is served through HTTPS in the deployment architecture. HSTS should not be enabled blindly on an HTTP development server.

## Logging rule

CEREBRO's security layer deliberately avoids logging request bodies. Application logs should never include:

- raw audio bytes
- full transcripts
- API keys
- WebSocket tokens
- stored personalization payloads

For production, place the application behind a reverse proxy/gateway that provides centralized access logging, TLS termination and, ideally, shared distributed rate limiting.

## Process-local limitations

The built-in rate limiter and live connection counter are process-local. In a multi-worker or horizontally scaled deployment, use a gateway/load balancer or shared store for global rate limiting and connection coordination.

## Threat model summary

CEREBRO's main application-level risks are:

| Risk | Current mitigation |
|---|---|
| Oversized uploads | endpoint limits + global body-size defense |
| Profile path traversal | strict profile-ID validation |
| Browser-origin abuse | restrictive CORS + WebSocket origin checks |
| Burst/API abuse | process-local rate limiter |
| Excess live sockets | per-IP connection cap |
| Raw audio persistence | not written by profile services |
| Stale personal data | automatic retention pruning |
| Forgotten profile state | one-call deletion endpoint |
| Secret leakage through URL | WebSocket token is sent in start message |
| Cached sensitive API responses | `Cache-Control: no-store` |

## What this is not

This is application-level hardening, not a full enterprise security program. A production deployment should still add:

- TLS termination
- centralized authentication/identity where appropriate
- shared rate limiting
- secret management
- infrastructure isolation
- monitoring/alerting
- backups and controlled restore procedures
- dependency vulnerability scanning
- regular penetration testing
