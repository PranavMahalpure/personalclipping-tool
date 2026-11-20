import httpx
import sys

BASE_URL = "http://127.0.0.1:8765"
HEADERS = {
    "Origin": BASE_URL,
    "Referer": f"{BASE_URL}/app"
}

def test_quick_switch():
    client = httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=10.0)
    
    # 1. Test /app HTML contains the Quick Switcher components
    r = client.get("/app")
    assert r.status_code == 200, f"/app returned {r.status_code}"
    html = r.text
    assert "ai-quickbar" in html, "ai-quickbar not in /app"
    assert "aiQuickBar" in html, "aiQuickBar function not in /app"
    assert "ainextmodel" in html, "ainextmodel handler not in /app"
    assert "aiquickselect" in html, "aiquickselect handler not in /app"
    assert "TOP_15_MODEL_LIST" in html, "TOP_15_MODEL_LIST not in /app"
    print("✓ /app frontend contains all AI quick switch UI components")

    # 2. Test /api/status contains enriched AI information
    r = client.get("/api/status")
    assert r.status_code == 200, f"/api/status returned {r.status_code}"
    st = r.json()
    ai_info = st.get("ai")
    assert ai_info, "No 'ai' in /api/status"
    assert "active_key_index" in ai_info, "active_key_index not in status.ai"
    assert "keys_count" in ai_info, "keys_count not in status.ai"
    assert ai_info["keys_count"] == 2, f"Expected 2 keys, got {ai_info['keys_count']}"
    assert "model" in ai_info, "model not in status.ai"
    assert "top_models" in ai_info, "top_models not in status.ai"
    assert len(ai_info["top_models"]) == 15, f"Expected 15 top models, got {len(ai_info['top_models'])}"
    print(f"✓ /api/status active model: {ai_info['model']}, active key index: {ai_info['active_key_index']}, keys count: {ai_info['keys_count']}")

    # 3. Test instant key switch
    init_key_idx = ai_info["active_key_index"]
    target_idx = 1 if init_key_idx == 0 else 0
    r = client.post("/api/ai/switch-key", json={"index": target_idx})
    assert r.status_code == 200, f"switch-key failed: {r.text}"
    data = r.json()
    assert data["active_key_index"] == target_idx
    
    # Verify status immediately reflects new key
    r_st = client.get("/api/status")
    assert r_st.json()["ai"]["active_key_index"] == target_idx, "Cache invalidation failed for switch-key"
    print(f"✓ Instant Key switch to Key {target_idx + 1} ({data['key_hint']}) verified in /api/status")

    # 4. Test instant model switch
    target_model = "gemini-3.5-flash-lite"
    r = client.post("/api/ai/switch-model", json={"model": target_model})
    assert r.status_code == 200, f"switch-model failed: {r.text}"
    assert r.json()["active_model"] == target_model
    
    # Verify status immediately reflects new model
    r_st2 = client.get("/api/status")
    assert r_st2.json()["ai"]["model"] == target_model, "Cache invalidation failed for switch-model"
    print(f"✓ Instant Model switch to {target_model} verified in /api/status")

    print("\nALL QUICK SWITCH CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_quick_switch()
