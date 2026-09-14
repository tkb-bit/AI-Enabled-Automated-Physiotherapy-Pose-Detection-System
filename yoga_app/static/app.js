/**
 * Yoga Studio - AI Physiotherapy & Movement Analysis Platform
 * Modern Frontend Telemetry & Interactivity Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  console.log('[+] Yoga Studio AI Platform initialized.');

  // Auto-dismiss toast notifications after 5 seconds
  const toasts = document.querySelectorAll('.toast');
  toasts.forEach(toast => {
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  });

  // Check if current page is the AI Assessment page
  if (document.getElementById('assessmentView')) {
    initAssessmentEngine();
  }
});

/**
 * Real-time AI Assessment telemetry loop, voice speech, and timer logic.
 */
function initAssessmentEngine() {
  const scoreElem = document.getElementById('hudFormScore');
  const scoreBar = document.getElementById('hudScoreBar');
  const confElem = document.getElementById('hudConfidence');
  const confBar = document.getElementById('hudConfBar');
  const repElem = document.getElementById('hudRepCount');
  const feedbackPrimary = document.getElementById('hudFeedbackPrimary');
  const feedbackSecondary = document.getElementById('hudFeedbackSecondary');
  const voiceToggle = document.getElementById('voiceToggle');
  const timerElem = document.getElementById('sessionTimer');
  const endSessionBtn = document.getElementById('endSessionBtn');

  let sessionSeconds = 0;
  let voiceEnabled = true;
  let lastSpokenText = "";
  let lastSpokenTime = 0;
  const voiceCooldownMs = 3500;

  if (voiceToggle) {
    voiceToggle.addEventListener('change', (e) => {
      voiceEnabled = e.target.checked;
    });
  }

  // Session duration timer
  const timerInterval = setInterval(() => {
    sessionSeconds++;
    const mins = String(Math.floor(sessionSeconds / 60)).padStart(2, '0');
    const secs = String(sessionSeconds % 60).padStart(2, '0');
    if (timerElem) timerElem.textContent = `${mins}:${secs}`;
  }, 1000);

  // Spoken feedback helper (Web Speech API)
  function speakFeedback(text) {
    if (!voiceEnabled || !text || !('speechSynthesis' in window)) return;
    
    const now = Date.now();
    if (text === lastSpokenText && (now - lastSpokenTime) < voiceCooldownMs) {
      return;
    }

    window.speechSynthesis.cancel(); // cancel pending speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
    
    lastSpokenText = text;
    lastSpokenTime = now;
  }

  const clientWebcam = document.getElementById('clientWebcam');
  const outputCanvas = document.getElementById('outputCanvas');
  const serverVideoFeed = document.getElementById('liveVideoFeed');
  const cameraModeText = document.getElementById('cameraModeText');

  let isSendingFrame = false;
  let clientCameraActive = false;
  let clientFrameInterval = null;

  function updateTelemetryUI(data) {
    if (!data) return;

    // Form Score
    const score = Math.round(data.form_score || 0);
    if (scoreElem) scoreElem.textContent = `${score}%`;
    if (scoreBar) scoreBar.style.width = `${score}%`;

    // Confidence
    const conf = Math.round(data.confidence || 0);
    if (confElem) confElem.textContent = `${conf}%`;
    if (confBar) confBar.style.width = `${conf}%`;

    // Rep Count
    if (repElem) repElem.textContent = data.rep_count || 0;

    // Feedback
    if (feedbackPrimary && data.primary_feedback) {
      feedbackPrimary.textContent = data.primary_feedback;
      if (data.primary_feedback.includes('✓')) {
        feedbackPrimary.style.color = 'var(--status-success)';
      } else if (data.primary_feedback.includes('⚠')) {
        feedbackPrimary.style.color = 'var(--status-warning)';
      } else {
        feedbackPrimary.style.color = 'var(--text-main)';
      }
    }

    if (feedbackSecondary && data.secondary_feedback) {
      if (data.secondary_feedback.length > 0) {
        feedbackSecondary.innerHTML = data.secondary_feedback.map(msg => `<li>${msg}</li>`).join('');
      } else {
        feedbackSecondary.innerHTML = '<li>Maintaining smooth motion.</li>';
      }
    }

    // Spoken Voice Feedback trigger
    if (data.voice_text) {
      speakFeedback(data.voice_text);
    }
  }

  // Attempt Client-Side Web Browser Webcam Capture
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia && clientWebcam) {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
      .then(stream => {
        clientWebcam.srcObject = stream;
        clientWebcam.play();
        clientCameraActive = true;

        if (cameraModeText) cameraModeText.textContent = 'CLIENT WEBCAM LIVE (HOSTED MODE)';
        if (serverVideoFeed) serverVideoFeed.style.display = 'none';
        if (outputCanvas) outputCanvas.style.display = 'block';

        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = 640;
        captureCanvas.height = 480;
        const captureCtx = captureCanvas.getContext('2d');

        clientFrameInterval = setInterval(() => {
          if (isSendingFrame || clientWebcam.paused || clientWebcam.ended) return;

          captureCtx.drawImage(clientWebcam, 0, 0, 640, 480);
          const base64Img = captureCanvas.toDataURL('image/jpeg', 0.5);
          isSendingFrame = true;

          fetch('/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Img })
          })
          .then(res => res.json())
          .then(data => {
            isSendingFrame = false;
            if (data && data.image && outputCanvas) {
              const img = new Image();
              img.onload = () => {
                outputCanvas.width = img.width;
                outputCanvas.height = img.height;
                const ctx = outputCanvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
              };
              img.src = data.image;
            }
            if (data && data.telemetry) {
              updateTelemetryUI(data.telemetry);
            }
          })
          .catch(err => {
            isSendingFrame = false;
          });
        }, 150);
      })
      .catch(err => {
        console.warn('[!] Client camera not allowed or unavailable. Using server fallback feed:', err);
        if (cameraModeText) cameraModeText.textContent = 'HOSTED DEMO FEED';
        if (serverVideoFeed) serverVideoFeed.style.display = 'block';
        if (outputCanvas) outputCanvas.style.display = 'none';
      });
  }

  // Telemetry polling fallback from Flask `/telemetry` endpoint every 300ms
  const telemetryInterval = setInterval(() => {
    if (clientCameraActive) return; // Skip polling if client webcam is active
    fetch('/telemetry')
      .then(res => res.json())
      .then(data => updateTelemetryUI(data))
      .catch(err => console.error('[!] Telemetry error:', err));
  }, 300);

  // End session button action
  if (endSessionBtn) {
    endSessionBtn.addEventListener('click', () => {
      clearInterval(timerInterval);
      clearInterval(telemetryInterval);

      const exerciseName = document.getElementById('assessmentView').dataset.exercise || 'Shoulder Rotation';
      const finalScore = parseFloat(scoreElem ? scoreElem.textContent : '90');
      const finalAccuracy = parseFloat(confElem ? confElem.textContent : '95');
      const finalReps = parseInt(repElem ? repElem.textContent : '0');
      const finalFeedback = feedbackPrimary ? feedbackPrimary.textContent : 'Completed session successfully.';

      endSessionBtn.disabled = true;
      endSessionBtn.textContent = 'Generating Session Report...';

      fetch('/complete_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exercise: exerciseName,
          form_score: finalScore,
          accuracy: finalAccuracy,
          repetitions: finalReps,
          duration: sessionSeconds,
          feedback: finalFeedback
        })
      })
      .then(res => res.json())
      .then(data => {
        if (data.redirect_url) {
          window.location.href = data.redirect_url;
        }
      })
      .catch(err => {
        alert('Failed to save session report.');
        endSessionBtn.disabled = false;
        endSessionBtn.textContent = 'End Session & Save';
      });
    });
  }
}
