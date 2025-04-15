# In src/ui/streamlit_app.py
# Make sure the login function matches the actual API URL
async def login(username: str, password: str) -> bool:
    """Login and get JWT token."""
    try:
        async with httpx.AsyncClient() as client:
            # Update to match the FastAPI router
            response = await client.post(
                f"{API_URL}/api/auth/token",
                data={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state["token"] = data["access_token"]
                st.session_state["username"] = username
                return True
            else:
                return False
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False
