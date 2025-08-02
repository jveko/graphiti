## Instructions for Using Graphiti's MCP Tools for Agent Memory

### Before Starting Any Task

- **Always search first:** Use the `search_nodes` tool to look for relevant preferences and procedures before beginning work.
- **Search for facts too:** Use the `search_facts` tool to discover relationships and factual information that may be relevant to your task.
- **Filter by entity type:** Specify `Preference`, `Procedure`, or `Requirement` in your node search to get targeted results.
  - ⚠️ **Important:** Entity filtering only works if the MCP server was started with `--use-custom-entities` flag enabled.
- **Review all matches:** Carefully examine any preferences, procedures, or facts that match your current task.

### Always Save New or Updated Information

- **Capture requirements and preferences immediately:** When a user expresses a requirement or preference, use `add_memory` to store it right away.
  - _Best practice:_ Split very long requirements into shorter, logical chunks.
- **Be explicit if something is an update to existing knowledge.** Only add what's changed or new to the graph.
- **Document procedures clearly:** When you discover how a user wants things done, record it as a procedure.
- **Record factual relationships:** When you learn about connections between entities, store these as facts.
- **Be specific with categories:** Label preferences and procedures with clear categories for better retrieval later.

### During Your Work

- **Respect discovered preferences:** Align your work with any preferences you've found.
- **Follow procedures exactly:** If you find a procedure for your current task, follow it step by step.
- **Apply relevant facts:** Use factual information to inform your decisions and recommendations.
- **Stay consistent:** Maintain consistency with previously identified preferences, procedures, and facts.

### Best Practices

- **Search before suggesting:** Always check if there's established knowledge before making recommendations.
- **Combine node and fact searches:** For complex tasks, search both nodes and facts to build a complete picture.
- **Use `center_node_uuid`:** When exploring related information, center your search around a specific node.
- **Prioritize specific matches:** More specific information takes precedence over general information.
- **Be proactive:** If you notice patterns in user behavior, consider storing them as preferences or procedures.

### Step-by-Step Workflow for Entity Filtering

#### Initial Setup (Required Once)
1. **Start server with custom entities enabled:**
   ```bash
   python mcp_server/graphiti_mcp_server_new.py --use-custom-entities
   ```

2. **Add data with entity extraction:**
   ```python
   add_memory(
       name="User Requirements", 
       episode_body="I require all reports to be generated in PDF format with executive summaries.",
       source="text"
   )
   ```

3. **Verify entities were extracted:**
   ```python
   list_node_labels()  # Check entity_type_counts for "Requirement", "Preference", "Procedure"
   ```

#### Daily Usage Workflow
1. **Search for specific entity types:**
   ```python
   search_memory_nodes(query="PDF reports", entity="Requirement")
   ```

2. **If no results, troubleshoot systematically:**
   ```python
   # Step 1: Check what entity types exist
   list_node_labels()
   
   # Step 2: Search without filter to see if nodes exist at all
   search_memory_nodes(query="PDF reports")
   
   # Step 3: Check specific entity type
   list_node_labels(entity_type="Requirement")
   ```

### Troubleshooting Entity Filtering

**Problem: Entity filtering returns no results**

**Solution Path:**
1. **Verify server configuration:** Use `list_node_labels()` - check `custom_entities_enabled` field
2. **Check existing data:** Look at `entity_type_counts` - are there any "Requirement", "Preference", "Procedure" entries?
3. **Historical data issue:** If counts show only "Untyped", your data was added before enabling custom entities
4. **Re-add critical data:** Add new episodes with entity extraction enabled

**Problem: Server returns configuration error**

**Solution:**
```bash
# Stop current server (Ctrl+C)
# Restart with correct flags:
python mcp_server/graphiti_mcp_server_new.py --use-custom-entities --group-id your_project_name
```

### Advanced Debugging Tools

- **`list_node_labels()`:** Overview of all nodes with entity type statistics
- **`list_node_labels(entity_type="Requirement")`:** Filter to specific entity type
- **`list_node_labels(limit=100)`:** Increase limit for larger datasets
- **`search_memory_nodes`:** Enhanced with detailed error messages and performance timing

### Performance Considerations

**Search Strategies (automatically selected):**
- **Cross-encoder:** Best quality, slower (~1-3s) - used when reranker available
- **RRF Hybrid:** Good quality, fast (~0.1-0.5s) - fallback when no reranker
- **Node Distance:** Proximity-based - used when `center_node_uuid` provided

**Tip:** Check server logs for search timing and strategy selection.

**Remember:** The knowledge graph is your memory. Use it consistently to provide personalized assistance that respects the user's established preferences, procedures, and factual context.
