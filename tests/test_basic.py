"""
Basic test to verify that we can pass a temporary vault to the indexer.
This is a placeholder test for now.
"""
import tempfile
from pathlib import Path


def test_temp_vault_creation():
    """Test that we can create and use a temporary vault directory."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple markdown file in the temp directory
        test_file = temp_path / "test.md"
        test_file.write_text("# Test Note\n\nThis is a test note.")
        
        # Verify the file exists
        assert test_file.exists()
        assert test_file.read_text() == "# Test Note\n\nThis is a test note."
        
        print(f"Created temporary vault at: {temp_path}")
        print("Basic vault creation test passed!")

if __name__ == "__main__":
    test_temp_vault_creation()