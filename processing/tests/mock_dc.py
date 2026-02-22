from processing.schemas import DocumentAnalysis

mock_doc_analysis = DocumentAnalysis(
    **{
        "structure_style": "Hierarchical Document Structure",
        "header_modifications": [
            {
                "header_name": "**Software Developer**",
                "modified_level": 2,
                "reason": "Header level incremented from H1 ('Experience') to H3, violating the 'increment by at most one' rule. It should be H2 as a direct subsection of 'Experience'.",
            },
            {
                "header_name": "_WellCare Insurance_",
                "modified_level": 3,
                "reason": "This header semantically describes the employer for the previous role, which should be level 2. Therefore, this header should be level 3.",
            },
            {
                "header_name": "**Software Developer**",
                "modified_level": 2,
                "reason": "This header represents a new role under 'Experience' and should be at the same level as the previous 'Software Developer' entry, which is H2.",
            },
            {
                "header_name": "Technical Skills",
                "modified_level": 1,
                "reason": "This header represents a new major section and should be H1, parallel to 'Experience' and 'Projects'.",
            },
            {
                "header_name": "Projects",
                "modified_level": 1,
                "reason": "This header represents a new major section and should be H1, parallel to 'Experience' and 'Projects'.",
            },
            {
                "header_name": "Education",
                "modified_level": 1,
                "reason": "This header represents a new major section and should be H1, parallel to 'Experience' and 'Projects'.",
            },
        ],
    }
)
