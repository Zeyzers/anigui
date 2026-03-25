# Watch‑Party Feature Design (Persistent P2P Session)

## 1️⃣ Overview
- **Goal**: Enable real‑time, synchronized watch‑party sessions where multiple Anigui users can watch the same anime episode together, stay in sync, and chat.
- **Key constraints**: Pure peer‑to‑peer (no persistent server), cross‑platform (Windows/Linux), minimal additional dependencies, secure (only AniList token is sensitive).

## 2️⃣ Architecture
- **WebRTC (aiortc)** is used for direct peer connections.
- **Signaling**: Host generates a short *session code* (JSON‑encoded ICE candidates + UUID) that participants paste to join. No external account‑based service; only free public STUN servers are used for NAT traversal.
- **Data channels**:
  - `control` (reliable, ordered) – carries sync, playback commands, and chat messages.
- **Encryption**: WebRTC traffic is encrypted by default (DTLS‑SRTP). No extra encryption needed for chat or video URLs.

## 3️⃣ Playback Synchronization (NTP‑style)
- Host polls MPV every **2 s** for playback position.
- Sends an NTP packet:
```json
{ "type": "sync", "t0": <host_send_utc>, "t1": <playback_seconds>, "t2": <host_recv_utc> }
```
- Peers compute round‑trip delay, adjust position, and **seek** if drift > 500 ms.
- Small drifts are ignored to avoid jitter.

## 4️⃣ Persistent Session Across Media Changes
- Host maintains `WatchPartySession` with `session_id`, `participants`, `current_media`, `is_active`.
- When the host selects a new episode/anime, `session.update_media(new_media)` broadcasts a `media_change` message:
```json
{ "type": "media_change", "media": { "url": "…", "title": "…", "start": 0 } }
```
- Peers stop current MPV instance, load the new URL, and seek to `start`. The NTP sync loop resumes automatically.
- UI shows **Current Episode** label that updates for all participants.
- Host‑only media control guarantees a single source of truth.

## 5️⃣ Chat Layer
- Simple JSON messages over the same `control` channel:
```json
{ "type": "chat", "nick": "Alice", "msg": "Hey!" }
```
- Displayed in a scrollable pane inside the watch‑party UI.

## 6️⃣ UI Integration
- New **Watch‑Party** tab with three sections:
  1. **Session controls** – *Create Party* (host) or *Join Party* (guest) with a text field for the session code and a *Copy to clipboard* button.
  2. **Participant list** – shows nicknames.
  3. **Chat pane** – message history + input.
- Host sees a **Switch Episode** button that triggers `media_change`.
- Guests see the current episode label but cannot change media.

## 7️⃣ Failure & Resync
| Situation | Action |
|---|---|
| Peer loses data channel | Attempt re‑handshake using original session code; on reconnection host sends a full sync packet. |
| Host leaves | Broadcast `session_end`; UI shows *Party ended*. |
| Network jitter > 2 s | Host sends `sync_full` to force immediate seek. |

## 8️⃣ Security & Privacy
- Only the AniList API token is ever sent over Anigui’s existing HTTPS calls; it is **not** transmitted over the watch‑party channel.
- WebRTC provides end‑to‑end encryption (DTLS‑SRTP) for all data channels.
- No persistent external service is required beyond free public STUN servers.

## 9️⃣ Decision Log
| Decision | Alternatives Considered | Rationale |
|---|---|---|
| Use **WebRTC (aiortc)** | libp2p, direct TCP sockets | WebRTC gives built‑in NAT traversal, encryption, and reliable data channels; mature Python bindings exist. |
| Sync via **NTP‑style timestamps** | Event‑driven play/pause only | NTP provides continuous drift correction, keeping playback tightly aligned across peers. |
| **Chat on same data channel** | Separate WebSocket server, external chat service | Simpler implementation; leverages existing DTLS encryption. |
| **Session code** for signaling | QR code, external signaling server | Text‑based copy‑paste works without accounts or extra infrastructure; easy to share. |
| **Persist session across media changes** | End‑of‑session per episode, new session per episode | Users want continuous group watching; `media_change` message updates all participants without needing a new session code. |

## 🔧 Next Steps (Implementation Tasks)
1. **Add Watch‑Party UI tab** – layout, session code input, participant list, chat pane, current episode label.
2. **Integrate aiortc peer handling** – create `WatchPartySession` class, signaling flow, data‑channel setup.
3. **Implement NTP sync loop** – host polling, packet format, peer drift correction.
4. **Add media_change handling** – host broadcast, peer reload, seamless episode switch.
5. **Testing & QA** – unit tests for sync logic, integration tests for peer connection, UI tests for session flow.

---
*Design created on 2026‑03‑25. This document should be kept in the repo and referenced during implementation.*
