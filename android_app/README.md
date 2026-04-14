# Viralytics Mobile

This is a minimal native Android client for the merged Viralytics / FashionSense backend.

It does three things:

1. captures a photo with the tablet camera,
2. uploads it to the PC backend at `/api/mobile/scan`,
3. sends refinement chat messages to `/api/chat`.

## What this app is

This is intentionally the simplest native Android path that can become a real APK quickly:

- native Android app,
- no WebView,
- no browser dependency,
- no streaming video yet,
- photo capture first, then refine with chat.

## Requirements

1. The PC server must be running on your local network.
2. The backend should be started with a LAN-visible host, for example:

```powershell
.\scripts\start_full_app.ps1 -BindHost 0.0.0.0 -BindPort 8000
```

3. The tablet and PC must be on the same Wi-Fi network.
4. Windows Firewall must allow inbound TCP on port `8000`.

## Android Studio

Open the `android_app/` folder in Android Studio.

Recommended:

- Android Studio Iguana or newer
- Android SDK 34
- JDK 17

Then let Gradle sync and run the app on the tablet.

## First run

In the app, set the server URL to your PC LAN IP:

```text
http://192.168.x.x:8000
```

You can find the PC IP with:

```powershell
ipconfig
```

Then:

1. tap `Capture Outfit`
2. take a photo
3. wait for detections and recommendations
4. type a refinement message
5. optionally enable `Replace vision result` to ignore the scan context

## Current limitations

- This app captures still photos, not live video.
- It uses the basic Android camera preview intent, not CameraX streaming.
- Chat history is not persisted yet.
- It is a prototype client, not a polished production app.

## Best next upgrade

If you want the next meaningful improvement, implement:

- CameraX live preview in-app
- session history/chat history persistence
- result cards instead of plain text
- server auto-discovery on LAN
